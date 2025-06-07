use std::borrow::Cow;

#[derive(Debug, Clone, Copy)]
pub enum PrimitiveType {
    Bool,
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

pub trait AsPrimitiveType {
    fn as_primitive_type() -> PrimitiveType;
}

impl AsPrimitiveType for u8 {
    fn as_primitive_type() -> PrimitiveType {
        PrimitiveType::U8
    }
}

#[derive(Debug)]
pub enum Primitive {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

pub struct TypeInfo<'a> {
    name: Cow<'a, str>,
}
