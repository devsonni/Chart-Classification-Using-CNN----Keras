/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Declarations                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {

// GPU Convolution backend configuration
class ConvolutionBackendConfig : public ::mlir::DictionaryAttr {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
  static ConvolutionBackendConfig get(
      ::mlir::IntegerAttr algorithm,
      ::mlir::BoolAttr tensor_ops_enabled,
      ::mlir::ArrayAttr knob_ids,
      ::mlir::ArrayAttr knob_values,
      ::mlir::BoolAttr is_cudnn_frontend,
      ::mlir::IntegerAttr workspace_size,
      ::mlir::ArrayAttr operand_0_layout,
      ::mlir::ArrayAttr operand_1_layout,
      ::mlir::ArrayAttr result_layout,
      ::mlir::MLIRContext* context);

  ::mlir::IntegerAttr algorithm() const;
  ::mlir::BoolAttr tensor_ops_enabled() const;
  ::mlir::ArrayAttr knob_ids() const;
  ::mlir::ArrayAttr knob_values() const;
  ::mlir::BoolAttr is_cudnn_frontend() const;
  ::mlir::IntegerAttr workspace_size() const;
  ::mlir::ArrayAttr operand_0_layout() const;
  ::mlir::ArrayAttr operand_1_layout() const;
  ::mlir::ArrayAttr result_layout() const;
};

} // namespace mlir
} // namespace lmhlo_gpu
