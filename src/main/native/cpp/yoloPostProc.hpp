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

#pragma once

#include <vector>

#include <jni.h>
#include <tensorflow/lite/c/c_api.h>

#include "utils.hpp"

/***
 * Performs YOLO post-processing including box decoding and NMS.
 * @param interpreter Pointer to the TensorFlow Lite interpreter.
 * @param boxThresh Confidence threshold for filtering boxes.
 * @param nmsThreshold IoU threshold for Non-Maximum Suppression.
 * @param env Pointer to the JNI environment.
 * @param input_img Pointer to the input OpenCV image matrix.
 * @return A JNI array of detection result objects.
 */
std::vector<detect_result_t> yoloPostProc(TfLiteInterpreter* interpreter,
                                          double boxThresh, double nmsThreshold,
                                          int input_img_width,
                                          int input_img_height);
