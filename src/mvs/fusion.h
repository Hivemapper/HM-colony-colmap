// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_MVS_FUSION_H_
#define COLMAP_SRC_MVS_FUSION_H_

#include <unordered_set>
#include <vector>
#include <string>
#include <map>

#include <Eigen/Core>

#include "mvs/depth_map.h"
#include "mvs/image.h"
#include "mvs/mat.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "mvs/workspace.h"
#include "util/alignment.h"
#include "util/cache.h"
#include "util/math.h"
#include "util/ply.h"
#include "util/threading.h"

namespace colmap {
namespace mvs {

struct FrameMetadata {
  int height;
  int width;
  std::string name;
};

struct FrameData {
  int height;
  int width;
  std::vector<int> coord2drow;
  std::vector<int> coord2dcol;
  std::vector<int> coord3dInd;
};

struct PointMetrics {
  size_t num_pixels = 0;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> nx;
  std::vector<float> ny;
  std::vector<float> nz;
  std::vector<float> px;
  std::vector<float> py;
  std::vector<float> pz;
  std::vector<uint8_t> r;
  std::vector<uint8_t> g;
  std::vector<uint8_t> b;
};

struct StereoFusionOptions {
  // Maximum image size in either dimension.
  int max_image_size = -1;

  // Minimum number of fused pixels to produce a point.
  int min_num_pixels = 5;

  // Maximum number of pixels to fuse into a single point.
  int max_num_pixels = 10000;

  // Maximum depth in consistency graph traversal.
  int max_traversal_depth = 100;

  // Maximum relative difference between measured and projected pixel.
  double max_reproj_error = 2.0f;

  // Maximum relative difference between measured and projected depth.
  double max_depth_error = 0.01f;

  // Maximum angular difference in degrees of normals of pixels to be fused.
  double max_normal_error = 10.0f;

  // Number of overlapping images to transitively check for fusing points.
  int check_num_images = 50;

  // Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
  // maps, normal maps, and consistency graphs of this number of images in
  // memory. A higher value leads to less disk access and faster fusion, while
  // a lower value leads to reduced memory usage. Note that a single image can
  // consume a lot of memory, if the consistency graph is dense.
  double cache_size = 32.0;

  // Output metrics
  bool coord_metrics = false;
  bool normal_metrics = false;
  bool view_ray_metrics = false;
  bool color_metrics = false;

  // Check the options for validity.
  bool Check() const;

  // Print the options to stdout.
  void Print() const;
};

class StereoFusion : public Thread {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoFusion(const StereoFusionOptions& options,
               const std::string& workspace_path,
               const std::string& workspace_format,
               const std::string& pmvs_option_name,
               const std::string& input_type);

  const std::vector<PlyPoint>& GetFusedPoints() const;
  const std::vector<PointMetrics>& GetFusedPointsMetrics() const;
  const std::vector<std::vector<int>>& GetFusedPointsVisibility() const;
  const std::map<int, FrameData>& Get2d3dCorrespondenceData() const;

 private:
  void Run();
  void Fuse(std::map<int, FrameMetadata> FrameMetadataMap);

  const StereoFusionOptions options_;
  const std::string workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::string input_type_;
  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;

  std::unique_ptr<Workspace> workspace_;
  std::vector<char> used_images_;
  std::vector<char> fused_images_;
  std::vector<std::vector<int>> overlapping_images_;
  std::vector<Mat<bool>> fused_pixel_masks_;
  std::vector<std::pair<int, int>> depth_map_sizes_;
  std::vector<std::pair<float, float>> bitmap_scales_;
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> P_;
  std::vector<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>> inv_P_;
  std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> inv_R_;
  std::vector<Eigen::Vector3f> C_;

  struct FusionData {
    int image_idx = kInvalidImageId;
    int row = 0;
    int col = 0;
    int traversal_depth = -1;
    bool operator()(const FusionData& data1, const FusionData& data2) {
      return data1.image_idx > data2.image_idx;
    }
  };

  // Next points to fuse.
  std::vector<FusionData> fusion_queue_;

  // Already fused points.
  std::vector<PlyPoint> fused_points_;
  std::vector<PointMetrics> fused_points_metrics_;
  std::vector<std::vector<int>> fused_points_visibility_;
  std::map<int, FrameData> frame_number_to_3dlist_;

  // Points of different pixels of the current point to be fused.
  PointMetrics fused_point_metric_;
  std::vector<float> fused_point_nx_;
  std::vector<float> fused_point_ny_;
  std::vector<float> fused_point_nz_;
  std::unordered_set<int> fused_point_visibility_;
  std::map<int, int> fused_point_visibility_row;
  std::map<int, int> fused_point_visibility_col;
};

// Write the visiblity information into a binary file of the following format:
//
//    <num_points : uint64_t>
//    <num_visible_images_for_point1 : uint32_t>
//    <point1_image_idx1 : uint32_t><point1_image_idx2 : uint32_t> ...
//    <num_visible_images_for_point2 : uint32_t>
//    <point2_image_idx2 : uint32_t><point2_image_idx2 : uint32_t> ...
//    ...
//
// Note that an image_idx in the case of the mvs::StereoFuser does not
// correspond to the image_id of a Reconstruction, but the index of the image in
// the mvs::Model, which is the location of the image in the images.bin/.txt.
void WritePointsVisibility(
    const std::string& path,
    const std::vector<std::vector<int>>& points_visibility);

void Write2d3dCorrespondenceData(
    const std::string& DataPath, const std::string& MetaDataPath,
    const std::map<int, FrameData>& correspondenceData);

void WriteFusedPointsMetrics(
  const std::string& path,
  const std::vector<PointMetrics>& points);

int getFrameNumberFromFilename(std::string& frameFileName);

}  // namespace mvs
}  // namespace colmap

#endif  // COLMAP_SRC_MVS_FUSION_H_
