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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef RUBIK_JNI_SRC_MAIN_NATIVE_INCLUDE_TENSORFLOW_RELEASE_VERSION_H_
#define RUBIK_JNI_SRC_MAIN_NATIVE_INCLUDE_TENSORFLOW_RELEASE_VERSION_H_

// A cc_library //third_party/tensorflow/core/public:release_version provides
// defines with the version data from //third_party/tensorflow/tf_version.bzl.
// The version suffix can be set by passing the build parameters
// --repo_env=ML_WHEEL_BUILD_DATE=<date> and
// --repo_env=ML_WHEEL_VERSION_SUFFIX=<suffix>.
// To update the project version, update tf_version.bzl.

#define _TF_STR_HELPER(x) #x
#define _TF_STR(x) _TF_STR_HELPER(x)

#ifndef TF_MAJOR_VERSION
#error "TF_MAJOR_VERSION is not defined!"
#endif

#ifndef TF_MINOR_VERSION
#error "TF_MINOR_VERSION is not defined!"
#endif

#ifndef TF_PATCH_VERSION
#error "TF_PATCH_VERSION is not defined!"
#endif

#ifndef TF_VERSION_SUFFIX
#error "TF_VERSION_SUFFIX is not defined!"
#endif

// e.g. "0.5.0" or "0.6.0-alpha".
#define TF_VERSION_STRING                                                      \
  (_TF_STR(TF_MAJOR_VERSION) "." _TF_STR(TF_MINOR_VERSION) "." _TF_STR(        \
      TF_PATCH_VERSION) TF_VERSION_SUFFIX)

#endif // RUBIK_JNI_SRC_MAIN_NATIVE_INCLUDE_TENSORFLOW_RELEASE_VERSION_H_
