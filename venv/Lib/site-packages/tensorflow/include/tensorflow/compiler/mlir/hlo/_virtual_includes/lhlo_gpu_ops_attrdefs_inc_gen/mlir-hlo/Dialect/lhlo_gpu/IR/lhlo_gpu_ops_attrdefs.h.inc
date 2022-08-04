/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ActivationAttr;
namespace detail {
struct ActivationAttrStorage;
} // namespace detail
class ActivationAttr : public ::mlir::Attribute::AttrBase<ActivationAttr, ::mlir::Attribute, detail::ActivationAttrStorage> {
public:
  using Base::Base;
public:
  static ActivationAttr get(::mlir::MLIRContext *context, ::mlir::lmhlo_gpu::Activation value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"activation"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::lmhlo_gpu::Activation getValue() const;
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ActivationAttr)

#endif  // GET_ATTRDEF_CLASSES

