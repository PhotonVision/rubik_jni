/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "obbPostProc.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <jni.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include "utils.hpp"

#ifndef NDEBUG
#define DEBUG_PRINT(...) std::printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) \
  do {                   \
  } while (0)
#endif

std::vector<detect_result_t> obbPostProc(TfLiteInterpreter* interpreter,
                                         double boxThresh, double nmsThreshold,
                                         int input_img_width,
                                         int input_img_height) {
  const TfLiteTensor* outputTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);

  const TfLiteQuantizationParams outputParams =
      TfLiteTensorQuantizationParams(outputTensor);

  const int numBoxes = TfLiteTensorDim(outputTensor, 1);
  DEBUG_PRINT("INFO: Detected %d boxes\n", numBoxes);

  // Debug tensor shapes
  DEBUG_PRINT("DEBUG: Boxes tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(outputTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(outputTensor, i));
  }
  std::printf("\n");
#endif

  if (TfLiteTensorType(outputTensor) != kTfLiteUInt8) {
    throw std::runtime_error("Expected uint8 tensor type");
  }

  uint8_t* outputData = static_cast<uint8_t*>(TfLiteTensorData(outputTensor));

  DEBUG_PRINT("DEBUG: Quantization params - output: zp=%d, scale=%f\n",
              outputParams.zero_point, outputParams.scale);

  std::vector<detect_result_t> candidateResults;

  DEBUG_PRINT("DEBUG: Image dimensions: %dx%d\n", input_img_width,
              input_img_height);

  for (int i = 0; i < numBoxes; ++i) {
    // Use proper dequantization for score
    float score =
        get_dequant_value(outputData, kTfLiteUInt8, i * 7 + 4,
                          outputParams.zero_point, outputParams.scale);
    if (score < boxThresh) {
      continue;
    }

    int classId = outputData[i * 7 + 5];

    // For tensor shape [1, 8400, 4], use sequential indexing per detection
    uint8_t raw_x_1_u8 = outputData[i * 7 + 0];
    uint8_t raw_y_1_u8 = outputData[i * 7 + 1];
    uint8_t raw_x_2_u8 = outputData[i * 7 + 2];
    uint8_t raw_y_2_u8 = outputData[i * 7 + 3];

    uint8_t raw_angle_u8 = outputData[i * 7 + 6];

    // Use proper dequantization for bbox coordinates (like we do for scores)
    float x1 = get_dequant_value(&raw_x_1_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y1 = get_dequant_value(&raw_y_1_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float x2 = get_dequant_value(&raw_x_2_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y2 = get_dequant_value(&raw_y_2_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);

    float angle =
        get_dequant_value(&raw_angle_u8, kTfLiteUInt8, 0,
                          outputParams.zero_point, outputParams.scale);

    float clamped_x1 =
        std::max(0.0f, std::min(x1, static_cast<float>(input_img_width)));
    float clamped_y1 =
        std::max(0.0f, std::min(y1, static_cast<float>(input_img_height)));
    float clamped_x2 =
        std::max(0.0f, std::min(x2, static_cast<float>(input_img_width)));
    float clamped_y2 =
        std::max(0.0f, std::min(y2, static_cast<float>(input_img_height)));

    float angle_degrees = angle * 180.0f / 3.14159265f;

    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

#ifndef NDEBUG
    if (candidateResults.size() < 5) {
      std::printf(
          " DEBUG: box %d - uint8 corners: (%d, %d) to (%d, %d), angle=%d\n", i,
          raw_x_1_u8, raw_y_1_u8, raw_x_2_u8, raw_y_2_u8, raw_angle_u8);
      std::printf(
          "DEBUG: box %d - dequantized corners: (%.2f, %.2f) to (%.2f, "
          "%.2f), angle=%.2f\n",
          i, x1, y1, x2, y2, angle);
      std::printf(
          "DEBUG: box %d - clamped corners: (%.2f, %.2f) to (%.2f, %.2f), "
          "score=%.3f, class=%d, angle=%.2f\n",
          i, clamped_x1, clamped_y1, clamped_x2, clamped_y2, score, classId,
          angle_degrees);
    }
#endif

    detect_result_t det;
    det.box.x1 = static_cast<int>(std::round(clamped_x1));
    det.box.y1 = static_cast<int>(std::round(clamped_y1));
    det.box.x2 = static_cast<int>(std::round(clamped_x2));
    det.box.y2 = static_cast<int>(std::round(clamped_y2));
    det.box.angle = angle_degrees;
    det.obj_conf = score;
    det.id = classId;

    candidateResults.push_back(det);
  }

  // NMS
  return optimizedNMS(candidateResults, static_cast<float>(nmsThreshold));
}
