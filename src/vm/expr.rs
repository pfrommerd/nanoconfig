use crate::array::Array;

use std::borrow::{Borrow, Cow};
use std::marker::PhantomData;

pub trait Context: Sized {
    type StringLiteral;
    type ArrayLiteral;

    type TreeRef: Borrow<Expr<Self>>;
    type ConsElems: IntoIterator<Item = Expr<Self>>;
    type ArgElems: IntoIterator<Item = Expr<Self>>;

    type VarRef: Borrow<str>;
    type OpRef: Borrow<str>;
    type LabelRef: Borrow<str>;
}

pub struct LocalContext<'ctx>(PhantomData<&'ctx ()>);

impl<'ctx> Context for LocalContext<'ctx> {
    // All of the internal storage types for things
    // that might be indirect references
    type StringLiteral = String;
    type ArrayLiteral = Array<'ctx>;

    type TreeRef = Box<Expr<Self>>;
    type ConsElems = Vec<Expr<Self>>;
    type ArgElems = Vec<Expr<Self>>;

    type VarRef = Cow<'ctx, str>;
    type OpRef = Cow<'ctx, str>;
    type LabelRef = Cow<'ctx, str>;
}

pub enum Literal<C: Context = LocalContext<'static>> {
    Array(C::ArrayLiteral),
    String(C::StringLiteral),
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u16),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i16),
    I64(i64),
}

pub struct Erase;
pub struct Dup;

pub struct Var<C: Context = LocalContext<'static>>(C::VarRef);
pub struct Label<C: Context = LocalContext<'static>>(C::LabelRef);
pub struct Cons<C: Context = LocalContext<'static>>(C::ConsElems);

pub struct Op<C: Context = LocalContext<'static>>(C::OpRef);
pub struct Args<C: Context = LocalContext<'static>>(C::ArgElems);

pub enum Expr<C: Context = LocalContext<'static>> {
    Literal(Literal<C>),
    Var(C::VarRef),
    Label(C::LabelRef),
    Erase,
    Cons(C::ConsElems),
    Dup,
    Op(C::OpRef),
    Args(C::ArgElems),
}
pub struct Redex<C: Context = LocalContext<'static>> {
    lhs: Expr<C>,
    rhs: Expr<C>,
}
pub struct Net<C: Context = LocalContext<'static>> {
    root: Expr<C>,
    redex: Vec<Redex<C>>,
}
