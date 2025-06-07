trait Format {}

struct SerdeFormat<F: Format> {
    format: F,
}
