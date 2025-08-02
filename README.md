Note:

To build the necessary tensorflow libraries, the following commands can be used

```
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/external:external_delegate
```
