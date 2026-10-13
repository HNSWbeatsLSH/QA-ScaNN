// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_
#define SCANN_BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/types.h"

namespace research_scann {
template <typename T>
class DenseDataset;
template <typename T>
class TypedDataset;
class KMeansTree;
template <typename T>
class SingleMachineSearcherBase;
class ScannConfig;

struct MultiTokenHashedDataset {
  vector<int64_t> offsets;

  vector<int32_t> tokens;

  unique_ptr<DenseDataset<uint8_t>> codes;

  StatusOr<DatapointPtr<uint8_t>> Get(DatapointIndex dp_idx,
                                      int32_t token) const {
    if (dp_idx + 1 >= offsets.size()) {
      return OutOfRangeError(
          absl::StrCat("Multi-token datapoint index out of range: ", dp_idx));
    }
    for (int64_t j = offsets[dp_idx]; j < offsets[dp_idx + 1]; ++j) {
      if (tokens[j] == token) return (*codes)[j];
    }
    return NotFoundError(absl::StrCat("Datapoint ", dp_idx,
                                      " has no multi-token code for token ",
                                      token, "."));
  }
};

struct SingleMachineFactoryOptions {
  SingleMachineFactoryOptions() = default;

  StatusOr<DimensionIndex> ComputeConsistentDimensionality(
      const ScannConfig& config, const Dataset* dataset = nullptr) const;

  TypeTag type_tag = kInvalidTypeTag;

  shared_ptr<vector<std::vector<DatapointIndex>>> datapoints_by_token;

  shared_ptr<PreQuantizedFixedPoint> pre_quantized_fixed_point;

  shared_ptr<DenseDataset<uint8_t>> hashed_dataset;

  shared_ptr<DenseDataset<uint8_t>> soar_hashed_dataset;

  shared_ptr<MultiTokenHashedDataset> multi_token_hashed_dataset;

  shared_ptr<DenseDataset<int16_t>> bfloat16_dataset;

  std::shared_ptr<CentersForAllSubspaces> ah_codebook;

  std::shared_ptr<SerializedPartitioner> serialized_partitioner;

  std::shared_ptr<const KMeansTree> kmeans_tree;

  std::shared_ptr<std::vector<std::string>> hash_parameters;

  shared_ptr<vector<int64_t>> crowding_attributes;
  shared_ptr<vector<std::string>> crowding_dimension_names;

  shared_ptr<ThreadPool> parallelization_pool;
};

}  // namespace research_scann

#endif
