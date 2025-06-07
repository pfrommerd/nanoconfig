use pyo3::conversion::FromPyObjectBound;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use pyo3::Bound;
use pyo3::{FromPyObject, PyObject};

use std::borrow::Borrow;
use std::fs::File;
use std::mem::ManuallyDrop;
use std::os::fd::{FromRawFd, RawFd};

pub struct PyFile {
    inner: PyObject,
    is_text: bool,
}
impl<'py> FromPyObject<'py> for PyFile {
    fn extract_bound(obj: &Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let text_io = consts::text_io_base(obj.py())?;
        let is_text_io = obj.is_instance(text_io)?;
        Ok(PyFile {
            inner: obj.clone().unbind(),
            is_text: is_text_io,
        })
    }
}
// The file object and a reference to the python object
// to prevent cleaning up of the file.
pub struct NativeFile(ManuallyDrop<File>, Option<PyObject>);
impl NativeFile {
    fn new(fd: RawFd, obj: Option<PyObject>) -> Self {
        NativeFile(unsafe { ManuallyDrop::new(File::from_raw_fd(fd)) }, obj)
    }
}

impl Borrow<File> for NativeFile {
    fn borrow(&self) -> &File {
        self.0.borrow()
    }
}

impl<'py> FromPyObject<'py> for NativeFile {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let fileno_str = consts::fileno(obj.py());
        if !obj.hasattr(fileno_str)? {
            return Err(PyTypeError::new_err(
                "Object does not have a fileno() method.",
            ));
        }
        let fd = obj
            .call_method0(consts::fileno(obj.py()))
            .expect("Object does not have a fileno() method.");
        let fd: RawFd = fd.extract()?;
        Ok(NativeFile::new(fd, Some(obj.clone().unbind())))
    }
}

// A variant which can parse any IO-esque type
#[derive(FromPyObject)]
pub enum PyIO {
    NativeFile(NativeFile),
    PythonFile(PyFile),
}

#[derive(FromPyObject, IntoPyObject)]
pub enum PyDataBuffer {
    Bytes(Py<PyBytes>),
    String(Py<PyString>),
}

mod consts {
    use pyo3::prelude::*;
    use pyo3::sync::GILOnceCell;
    use pyo3::types::PyString;
    use pyo3::{intern, Bound, Py, PyResult, Python};

    pub fn fileno(py: Python) -> &Bound<PyString> {
        intern!(py, "fileno")
    }
    pub fn read(py: Python) -> &Bound<PyString> {
        intern!(py, "read")
    }
    pub fn write(py: Python<'_>) -> &Bound<PyString> {
        intern!(py, "write")
    }
    pub fn seek(py: Python<'_>) -> &Bound<PyString> {
        intern!(py, "seek")
    }
    pub fn flush(py: Python<'_>) -> &Bound<PyString> {
        intern!(py, "flush")
    }
    pub fn text_io_base(py: Python) -> PyResult<&Bound<PyAny>> {
        static INSTANCE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
        INSTANCE
            .get_or_try_init(py, || {
                let io = PyModule::import(py, "io")?;

                let cls = io.getattr("TextIOBase")?;

                Ok(cls.unbind())
            })
            .map(|x| x.bind(py))
    }
}
