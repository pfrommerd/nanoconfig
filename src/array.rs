use std::borrow::{Borrow, Cow};
use std::marker::PhantomData;

pub enum Dim {
    Fixed(usize),
    Var(String),
    Jagged,
}
pub struct Shape(Vec<Dim>);
pub enum ArrayImpl {}

// The array type actually owns the
// memory of an array
pub struct Array<'ctx>(PhantomData<&'ctx ()>);

impl<'a, 'ctx> Borrow<ArrayView<'a, 'ctx>> for Array<'ctx> {
    fn borrow(&self) -> &ArrayView<'a, 'ctx> {
        todo!()
    }
}

// A cheap-to-clone view into an array.
pub struct ArrayView<'a, 'ctx>(PhantomData<&'a Array<'ctx>>);

impl<'a, 'ctx> ToOwned for ArrayView<'a, 'ctx> {
    type Owned = Array<'ctx>;
    fn to_owned(&self) -> Self::Owned {
        todo!()
    }
}

pub type CowArray<'a, 'ctx> = Cow<'a, ArrayView<'a, 'ctx>>;
