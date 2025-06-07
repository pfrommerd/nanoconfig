use crate::array::{Array, ArrayView, CowArray, Shape};
use crate::types::{AsPrimitiveType, Primitive, PrimitiveType, TypeInfo};
use std::borrow::{Borrow, Cow};
use uuid::Uuid;

// A Ref is anything that is Borrowable as a type T
// and has a unique uuid.
// NOTE: Uuid's are not stable between serialization/deserialization
// such that an object can be deserialized multiple time.
pub trait Ref<T>: Borrow<T> {
    fn id(self) -> Uuid;
}

// We have visitors for introspecting the data model itself,
// for visiting a wealky owned version of the data
pub trait ModelVisitor<Leaf> {
    type Output;
    fn visited_primitive(self, ty: PrimitiveType) -> Self::Output;
    fn visited_leaf_ref(self);
}

pub trait Visitor<Leaf> {
    type Output;
    type LeafRef: ToOwned<Owned = Leaf>;

    fn visited_primitive(self, prim: Primitive) -> Self::Output;
    fn visited_leaf(self, leaf: Self::LeafRef) -> Self::Output;
}

pub trait IntoVisitor<Leaf> {
    type Output;

    fn visited_primitive(self, prim: Primitive) -> Self::Output;
    fn visited_leaf(self, leaf: Leaf) -> Self::Output;
}

// Graph models, GraphSerialize, and GraphDeserialize
pub trait GraphModel<Leaf> {
    fn visit_model<V: ModelVisitor<Leaf>>(&self, v: V) -> V::Output;
}

pub trait GraphSerialize<Leaf>: GraphModel<Leaf> {
    fn visit<R, V>(&self, v: V) -> V::Output
    where
        R: ToOwned<Owned = Leaf>,
        V: Visitor<R>;

    fn visit_into<R, V>(self, v: V) -> V::Output
    where
        R: ToOwned<Owned = Leaf>,
        V: IntoVisitor<R>;
}

pub trait GraphSerializer<Leaf> {
    type Error;

    fn serialize(&mut self, g: &impl GraphSerialize<Leaf>) -> Result<(), Self::Error>;
    fn serialize_into(&mut self, g: impl GraphSerialize<Leaf>) -> Result<(), Self::Error>;
}

pub trait GraphDeserializer<Leaf> {
    type Err;

    fn deserialize<V: Visitor<Leaf>>(
        &mut self,
        t: &impl GraphModel<Leaf>,
        v: V,
    ) -> Result<V::Output, Self::Err>;
}

trait GraphDeserialize<Leaf> {
    fn
}

// pub trait EnumModel<'ctx> {
//     fn num_variants(&self) -> usize;
//     fn variant_name(&self, i: usize) -> Option<&str>;
//     fn visit_variant<V: ModelVisitor<'ctx>>(&self, i: usize, v: V) -> V::Output;
// }

// // Serialization types
// pub trait ContainerSerialize<'ctx>: ContainerModel<'ctx> + Sized {
//     fn serialize_children<S: GraphSerializer<'ctx>>(&mut self, s: &mut S) -> Result<(), S::Err>;
//     fn serialize_children_into<S: GraphSerializer<'ctx>>(self, s: &mut S) -> Result<(), S::Err>;
// }
// pub trait VariantSerialize<'ctx>: EnumModel<'ctx> {
//     fn variant_type(&self) -> usize;
//     fn serialize_data<S: GraphSerializer<'ctx>>(&mut self, s: &mut S) -> Result<(), S::Err>;
// }

// pub trait GraphSerializer<'ctx> {
//     type Output;
//     type Err;

//     fn serialize_prim<T: Into<Primitive>>(&mut self, prim: T) -> Result<(), Self::Err>;

//     // Allows the serializer to assume ownership of the values
//     fn serialize_string<T: Into<String>>(&mut self, t: T) -> Result<(), Self::Err>;
//     fn serialize_bytes<T: Into<Vec<u8>>>(&mut self, t: T) -> Result<(), Self::Err>;
//     fn serialize_array<T: Into<Array<'ctx>>>(&mut self, t: T) -> Result<(), Self::Err>;

//     // Makes a copy of the values
//     fn serialize_str<T: AsRef<str>>(&mut self, t: T) -> Result<(), Self::Err>;
//     fn serialize_bytes_slice<T: AsRef<[u8]>>(&mut self, t: &T) -> Result<(), Self::Err>;
//     fn serialize_array_view<T>(&mut self, t: T) -> Result<(), Self::Err>
//     where
//         for<'a> T: Into<ArrayView<'a, 'ctx>>;

//     // Serialize a generic map (aka dictionary)
//     fn serialize_map(&mut self, fields: impl ContainerSerialize<'ctx>) -> Result<(), Self::Err>;
//     fn serialize_struct(
//         &mut self,
//         typ: &TypeInfo,
//         fields: impl ContainerSerialize<'ctx>,
//     ) -> Result<(), Self::Err>;

//     fn serialize_list(&mut self, entries: impl ContainerSerialize<'ctx>) -> Result<(), Self::Err>;
//     fn serialize_tuple(&mut self, entries: impl ContainerSerialize<'ctx>) -> Result<(), Self::Err>;
//     fn serialize_named_tuple(
//         &mut self,
//         typ: &TypeInfo,
//         entries: impl ContainerSerialize<'ctx>,
//     ) -> Result<(), Self::Err>;

//     fn serialize_variant(&mut self, variant: impl VariantSerialize<'ctx>) -> Result<(), Self::Err>;

//     // Small wrappers
//     fn serialize_newtype_struct(
//         &mut self,
//         typ: &TypeInfo,
//         payload: impl GraphSerializable<'ctx>,
//     ) -> Result<(), Self::Err>;
//     fn serialize_newtype_variant(
//         &mut self,
//         typ: &TypeInfo,
//         variant_name: &str,
//         payload: impl GraphSerializable<'ctx>,
//     ) -> Result<(), Self::Err>;

//     fn serialize_ref<T>(&mut self, r: impl Ref<T>) -> Result<(), Self::Err>
//     where
//         T: GraphSerializable<'ctx>;

//     fn finish(self) -> Result<Self::Output, Self::Err>;
// }

// pub trait GraphSerializable<'ctx> {
//     fn serialize<S>(&self, s: S)
//     where
//         S: GraphSerializer<'ctx>;

//     fn serialize_into<S>(self, s: S)
//     where
//         S: GraphSerializer<'ctx>;
// }

// // Deserialization!

// pub trait ContainerDeserialize<'buf, 'ctx> {
//     fn num_children_hint(&self) -> Option<usize>;
//     fn next_child(&mut self) -> Option<impl GraphDeserializer>;
//     // Returns key, value pair
//     fn next_entry(&mut self) -> Option<(Cow<'buf, str>, impl GraphDeserializer)>;
// }

// pub trait GraphDeserializer<'buf, 'ctx> {
//     type Err;
//     // Primitive deserialization
//     fn deserialize_prim(&mut self, prim_type: PrimitiveType) -> Result<Primitive, Self::Err>;
//     // A helper function to deserialize a primitive
//     fn deserialize_prim_as<T: AsPrimitiveType + TryFrom<Primitive>>(
//         &mut self,
//     ) -> Result<T, Self::Err> {
//         let ty: PrimitiveType = T::as_primitive_type();
//         let r: Primitive = self.deserialize_prim(ty)?;
//         match r.try_into() {
//             Ok(s) => Ok(s),
//             Err(_) => panic!("AsPrimitiveType failed TryFrom for {ty:?}"),
//         }
//     }

//     fn deserialize_string(&mut self) -> Result<Cow<'buf, str>, Self::Err>;
//     fn deserialize_bytes(&mut self) -> Result<Cow<'buf, [u8]>, Self::Err>;
//     fn deserialize_array(&mut self) -> Result<CowArray<'buf, 'ctx>, Self::Err>;

//     fn deserialize_map(&mut self) -> () {
//         panic!()
//     }

//     fn deserialize_ref<R, T>(&mut self) -> Result<T, Self::Err>
//     where
//         R: Ref<T>,
//         T: GraphDeserialize<'buf, 'ctx>;
// }
