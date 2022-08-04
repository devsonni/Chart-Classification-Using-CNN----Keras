/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace lmhlo_gpu {
class AllReduceDoneOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class AllReduceStartOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class CholeskyOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardFilterOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvBackwardInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardFusedSideInputOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class ConvForwardOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMMOp;
} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {
class GEMM_BiasOp;
} // namespace lmhlo_gpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceDoneOp declarations
//===----------------------------------------------------------------------===//

class AllReduceDoneOpAdaptor {
public:
  AllReduceDoneOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AllReduceDoneOpAdaptor(AllReduceDoneOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::ValueRange operands();
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllReduceDoneOp : public ::mlir::Op<AllReduceDoneOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceDoneOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_done");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::Operation::operand_range operands();
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange tokenMutable();
  ::mlir::MutableOperandRange operandsMutable();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::ValueRange operands, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::ValueRange operands, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceDoneOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::AllReduceStartOp declarations
//===----------------------------------------------------------------------===//

class AllReduceStartOpAdaptor {
public:
  AllReduceStartOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AllReduceStartOpAdaptor(AllReduceStartOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr replica_groupsAttr();
  ::mlir::DenseIntElementsAttr replica_groups();
  ::mlir::BoolAttr constrain_layoutAttr();
  bool constrain_layout();
  ::mlir::mhlo::ChannelHandle channel_idAttr();
  ::llvm::Optional<::mlir::mhlo::ChannelHandle> channel_id();
  ::mlir::BoolAttr use_global_device_idsAttr();
  bool use_global_device_ids();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &computation();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AllReduceStartOp : public ::mlir::Op<AllReduceStartOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceStartOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("replica_groups"), ::llvm::StringRef("constrain_layout"), ::llvm::StringRef("channel_id"), ::llvm::StringRef("use_global_device_ids")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr replica_groupsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr replica_groupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr constrain_layoutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr constrain_layoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr channel_idAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr channel_idAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr use_global_device_idsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr use_global_device_idsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.all_reduce_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange operandsMutable();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value token();
  ::mlir::Region &computation();
  ::mlir::DenseIntElementsAttr replica_groupsAttr();
  ::mlir::DenseIntElementsAttr replica_groups();
  ::mlir::BoolAttr constrain_layoutAttr();
  bool constrain_layout();
  ::mlir::mhlo::ChannelHandle channel_idAttr();
  ::llvm::Optional<::mlir::mhlo::ChannelHandle> channel_id();
  ::mlir::BoolAttr use_global_device_idsAttr();
  bool use_global_device_ids();
  void replica_groupsAttr(::mlir::DenseIntElementsAttr attr);
  void constrain_layoutAttr(::mlir::BoolAttr attr);
  void channel_idAttr(::mlir::mhlo::ChannelHandle attr);
  void use_global_device_idsAttr(::mlir::BoolAttr attr);
  ::mlir::Attribute removeChannel_idAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, ::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, ::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, ::mlir::BoolAttr constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, ::mlir::BoolAttr use_global_device_ids);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ValueRange results, ::mlir::DenseIntElementsAttr replica_groups, bool constrain_layout, /*optional*/::mlir::mhlo::ChannelHandle channel_id, bool use_global_device_ids = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::AllReduceStartOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp declarations
//===----------------------------------------------------------------------===//

class CholeskyOpAdaptor {
public:
  CholeskyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CholeskyOpAdaptor(CholeskyOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr is_lowerAttr();
  bool is_lower();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CholeskyOp : public ::mlir::Op<CholeskyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CholeskyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("is_lower")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr is_lowerAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr is_lowerAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.cholesky");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::Value info();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  ::mlir::MutableOperandRange infoMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr is_lowerAttr();
  bool is_lower();
  void is_lowerAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_lower);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::CholeskyOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardFilterOpAdaptor {
public:
  ConvBackwardFilterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardFilterOp : public ::mlir::Op<ConvBackwardFilterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardFilterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardfilter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value d_output();
  ::mlir::Value d_filter();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange d_filterMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardFilterOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp declarations
//===----------------------------------------------------------------------===//

class ConvBackwardInputOpAdaptor {
public:
  ConvBackwardInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvBackwardInputOpAdaptor(ConvBackwardInputOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvBackwardInputOp : public ::mlir::Op<ConvBackwardInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvBackwardInputOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_backwardinput");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value d_output();
  ::mlir::Value filter();
  ::mlir::Value d_input();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange d_outputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange d_inputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvBackwardInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedOpAdaptor {
public:
  ConvForwardFusedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardFusedOpAdaptor(ConvForwardFusedOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ActivationAttr activation_modeAttr();
  ::mlir::lmhlo_gpu::Activation activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedOp : public ::mlir::Op<ConvForwardFusedOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<5>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("backend_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr activation_modeAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr backend_configAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ActivationAttr activation_modeAttr();
  ::mlir::lmhlo_gpu::Activation activation_mode();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 12 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardFusedSideInputOpAdaptor {
public:
  ConvForwardFusedSideInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardFusedSideInputOpAdaptor(ConvForwardFusedSideInputOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ActivationAttr activation_modeAttr();
  ::mlir::lmhlo_gpu::Activation activation_mode();
  ::mlir::FloatAttr side_input_scaleAttr();
  ::llvm::APFloat side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardFusedSideInputOp : public ::mlir::Op<ConvForwardFusedSideInputOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardFusedSideInputOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("activation_mode"), ::llvm::StringRef("side_input_scale"), ::llvm::StringRef("backend_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr activation_modeAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr activation_modeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  ::mlir::StringAttr side_input_scaleAttrName() {
    return getAttributeNameForIndex(11);
  }

  static ::mlir::StringAttr side_input_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 11);
  }

  ::mlir::StringAttr backend_configAttrName() {
    return getAttributeNameForIndex(12);
  }

  static ::mlir::StringAttr backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 12);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward_fused_with_side_input");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value bias();
  ::mlir::Value side_input();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ActivationAttr activation_modeAttr();
  ::mlir::lmhlo_gpu::Activation activation_mode();
  ::mlir::FloatAttr side_input_scaleAttr();
  ::llvm::APFloat side_input_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::lmhlo_gpu::ActivationAttr attr);
  void side_input_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ActivationAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::Activation activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 13 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardFusedSideInputOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp declarations
//===----------------------------------------------------------------------===//

class ConvForwardOpAdaptor {
public:
  ConvForwardOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConvForwardOpAdaptor(ConvForwardOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ConvForwardOp : public ::mlir::Op<ConvForwardOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConvForwardOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("window_strides"), ::llvm::StringRef("padding"), ::llvm::StringRef("lhs_dilation"), ::llvm::StringRef("rhs_dilation"), ::llvm::StringRef("window_reversal"), ::llvm::StringRef("dimension_numbers"), ::llvm::StringRef("feature_group_count"), ::llvm::StringRef("batch_group_count"), ::llvm::StringRef("precision_config"), ::llvm::StringRef("result_scale"), ::llvm::StringRef("backend_config")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr window_stridesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr window_stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr paddingAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr paddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr lhs_dilationAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr lhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr rhs_dilationAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr rhs_dilationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr window_reversalAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr window_reversalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr dimension_numbersAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr feature_group_countAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr feature_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr batch_group_countAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr batch_group_countAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr precision_configAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr precision_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr result_scaleAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr result_scaleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr backend_configAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr backend_configAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.conv_forward");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Value output();
  ::mlir::Value scratch();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange scratchMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseIntElementsAttr window_stridesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > window_strides();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  ::mlir::DenseIntElementsAttr lhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > lhs_dilation();
  ::mlir::DenseIntElementsAttr rhs_dilationAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > rhs_dilation();
  ::mlir::DenseElementsAttr window_reversalAttr();
  ::llvm::Optional< ::mlir::DenseElementsAttr > window_reversal();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbersAttr();
  ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers();
  ::mlir::IntegerAttr feature_group_countAttr();
  uint64_t feature_group_count();
  ::mlir::IntegerAttr batch_group_countAttr();
  uint64_t batch_group_count();
  ::mlir::ArrayAttr precision_configAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > precision_config();
  ::mlir::FloatAttr result_scaleAttr();
  ::llvm::APFloat result_scale();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_configAttr();
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config();
  void window_stridesAttr(::mlir::DenseIntElementsAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  void lhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void rhs_dilationAttr(::mlir::DenseIntElementsAttr attr);
  void window_reversalAttr(::mlir::DenseElementsAttr attr);
  void dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbersAttr attr);
  void feature_group_countAttr(::mlir::IntegerAttr attr);
  void batch_group_countAttr(::mlir::IntegerAttr attr);
  void precision_configAttr(::mlir::ArrayAttr attr);
  void result_scaleAttr(::mlir::FloatAttr attr);
  void backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr);
  ::mlir::Attribute removeWindow_stridesAttr();
  ::mlir::Attribute removePaddingAttr();
  ::mlir::Attribute removeLhs_dilationAttr();
  ::mlir::Attribute removeRhs_dilationAttr();
  ::mlir::Attribute removeWindow_reversalAttr();
  ::mlir::Attribute removePrecision_configAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, /*optional*/::mlir::DenseElementsAttr window_reversal, ::mlir::mhlo::ConvDimensionNumbersAttr dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::ConvForwardOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp declarations
//===----------------------------------------------------------------------===//

class GEMMOpAdaptor {
public:
  GEMMOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GEMMOpAdaptor(GEMMOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMMOp : public ::mlir::Op<GEMMOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMMOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("batch_size"), ::llvm::StringRef("lhs_stride"), ::llvm::StringRef("rhs_stride"), ::llvm::StringRef("algorithm")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dot_dimension_numbersAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dot_dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr alpha_realAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr alpha_realAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr alpha_imagAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr alpha_imagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr batch_sizeAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr batch_sizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr lhs_strideAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr lhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr rhs_strideAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr rhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr algorithmAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr algorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void lhs_strideAttr(::mlir::IntegerAttr attr);
  void rhs_strideAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMMOp)

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMM_BiasOp declarations
//===----------------------------------------------------------------------===//

class GEMM_BiasOpAdaptor {
public:
  GEMM_BiasOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GEMM_BiasOpAdaptor(GEMM_BiasOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::FloatAttr betaAttr();
  ::llvm::APFloat beta();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class GEMM_BiasOp : public ::mlir::Op<GEMM_BiasOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GEMM_BiasOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dot_dimension_numbers"), ::llvm::StringRef("alpha_real"), ::llvm::StringRef("alpha_imag"), ::llvm::StringRef("beta"), ::llvm::StringRef("batch_size"), ::llvm::StringRef("lhs_stride"), ::llvm::StringRef("rhs_stride"), ::llvm::StringRef("algorithm")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dot_dimension_numbersAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dot_dimension_numbersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr alpha_realAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr alpha_realAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr alpha_imagAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr alpha_imagAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr betaAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr betaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr batch_sizeAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr batch_sizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr lhs_strideAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr lhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr rhs_strideAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr rhs_strideAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr algorithmAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr algorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lmhlo_gpu.gemm_bias");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value lhs();
  ::mlir::Value rhs();
  ::mlir::Value bias();
  ::mlir::Value output();
  ::mlir::MutableOperandRange lhsMutable();
  ::mlir::MutableOperandRange rhsMutable();
  ::mlir::MutableOperandRange biasMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbersAttr();
  ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers();
  ::mlir::FloatAttr alpha_realAttr();
  ::llvm::APFloat alpha_real();
  ::mlir::FloatAttr alpha_imagAttr();
  ::llvm::APFloat alpha_imag();
  ::mlir::FloatAttr betaAttr();
  ::llvm::APFloat beta();
  ::mlir::IntegerAttr batch_sizeAttr();
  uint64_t batch_size();
  ::mlir::IntegerAttr lhs_strideAttr();
  uint64_t lhs_stride();
  ::mlir::IntegerAttr rhs_strideAttr();
  uint64_t rhs_stride();
  ::mlir::IntegerAttr algorithmAttr();
  ::llvm::Optional<uint64_t> algorithm();
  void dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbersAttr attr);
  void alpha_realAttr(::mlir::FloatAttr attr);
  void alpha_imagAttr(::mlir::FloatAttr attr);
  void betaAttr(::mlir::FloatAttr attr);
  void batch_sizeAttr(::mlir::IntegerAttr attr);
  void lhs_strideAttr(::mlir::IntegerAttr attr);
  void rhs_strideAttr(::mlir::IntegerAttr attr);
  void algorithmAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeAlgorithmAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::mlir::FloatAttr alpha_real, ::mlir::FloatAttr alpha_imag, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr lhs_stride, ::mlir::IntegerAttr rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers, ::llvm::APFloat alpha_real, ::llvm::APFloat alpha_imag, ::llvm::APFloat beta, uint64_t batch_size, uint64_t lhs_stride, uint64_t rhs_stride, /*optional*/::mlir::IntegerAttr algorithm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace lmhlo_gpu
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::lmhlo_gpu::GEMM_BiasOp)


#endif  // GET_OP_CLASSES

