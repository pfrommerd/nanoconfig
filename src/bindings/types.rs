use crate::vm::array::Shape;
use std::sync::Arc;

struct StructField {}

struct StructType {
    fields: Vec<StructField>,
}

// Represents all possible datatypes
enum DataType {
    PyCustom(Arc<PyCustomType>),
    // The struct type and handle to underlying python object
    Dataclass(Arc<StructType>),
    Tuple(Vec<DataType>),
    List(Vec<DataType>),
    Dict(Vec<String, DataType>),

    Array(Option<Box<DataType>>, Option<Shape>),
    Bool,
    Bytes,
    String,

    I8,
    I16,
    I32,
    I64,

    U8,
    U16,
    U32,
    U64,

    F32,
    F64,
}

enum Data {}
