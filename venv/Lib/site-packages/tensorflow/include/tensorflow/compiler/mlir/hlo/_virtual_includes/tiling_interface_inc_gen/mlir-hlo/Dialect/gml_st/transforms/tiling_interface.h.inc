/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gml_st {
class TilingInterface;
namespace detail {
struct TilingInterfaceInterfaceTraits {
  struct Concept {
    ::mlir::Value (*tile)(const Concept *impl, ::mlir::Operation *, ::mlir::gml_st::MaterializeOp, OpBuilder&);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gml_st::TilingInterface;
    Model() : Concept{tile} {}

    static inline ::mlir::Value tile(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gml_st::TilingInterface;
    FallbackModel() : Concept{tile} {}

    static inline ::mlir::Value tile(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    ::mlir::Value tile(::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder&builder) const;
  };
};template <typename ConcreteOp>
struct TilingInterfaceTrait;

} // namespace detail
class TilingInterface : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TilingInterfaceTrait<ConcreteOp> {};
  ::mlir::Value tile(::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder);
};
namespace detail {
  template <typename ConcreteOp>
  struct TilingInterfaceTrait : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    ::mlir::Value tile(::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder) {
      return {};
    }
  };
}// namespace detail
template<typename ConcreteOp>
::mlir::Value detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::tile(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).tile(materialize, builder);
}
template<typename ConcreteOp>
::mlir::Value detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::tile(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder& builder) {
  return static_cast<const ConcreteOp *>(impl)->tile(tablegen_opaque_val, materialize, builder);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Value detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::tile(::mlir::Operation *tablegen_opaque_val, ::mlir::gml_st::MaterializeOp materialize, OpBuilder&builder) const {
return {};
}
} // namespace gml_st
} // namespace mlir
