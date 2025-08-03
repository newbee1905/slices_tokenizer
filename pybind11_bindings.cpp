#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "slices_tokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(slices_tokenizer, m) {
	m.doc() = "Fast C++ SLICES Tokenizer with Python bindings";
	
	// Bind TokenSpan struct
	py::class_<TokenSpan>(m, "TokenSpan")
		.def(py::init<const std::string&, size_t, size_t>())
		.def_readwrite("token", &TokenSpan::token)
		.def_readwrite("start", &TokenSpan::start)
		.def_readwrite("end", &TokenSpan::end)
		.def("__repr__", [](const TokenSpan& span) {
			return "TokenSpan(token='" + span.token + "', start=" + 
				   std::to_string(span.start) + ", end=" + std::to_string(span.end) + ")";
		});
	
	// Bind Encoding struct
	py::class_<Encoding>(m, "Encoding")
		.def(py::init<>())
		.def_readwrite("tokens", &Encoding::tokens)
		.def_readwrite("ids", &Encoding::ids)
		.def_readwrite("spans", &Encoding::spans)
		.def("clear", &Encoding::clear)
		.def("__len__", [](const Encoding& enc) { return enc.tokens.size(); })
		.def("__repr__", [](const Encoding& enc) {
			return "Encoding(tokens=" + std::to_string(enc.tokens.size()) + 
				   ", ids=" + std::to_string(enc.ids.size()) + ")";
		});
	
	// Bind main SLICESTokenizer class
	py::class_<SLICESTokenizer>(m, "SLICESTokenizer")
		.def(py::init<>())
		.def("pre_tokenize", &SLICESTokenizer::pre_tokenize,
			 "Pre-tokenize a string into SLICES components",
			 py::arg("text"))
		.def("train_from_iterator", &SLICESTokenizer::train_from_iterator,
			 "Train the tokenizer from a list of strings",
			 py::arg("training_data"), py::arg("vocab_size") = 1000)
		.def("encode", &SLICESTokenizer::encode,
			 "Encode a string to token IDs",
			 py::arg("text"))
		.def("decode", &SLICESTokenizer::decode,
			 "Decode token IDs back to string",
			 py::arg("ids"))
		.def("token_to_id", &SLICESTokenizer::token_to_id,
			 "Get token ID for a specific token",
			 py::arg("token"))
		.def("id_to_token", &SLICESTokenizer::id_to_token,
			 "Get token for a specific ID",
			 py::arg("id"))
		.def("get_vocab_size", &SLICESTokenizer::get_vocab_size,
			 "Get vocabulary size")
		.def("get_vocab", &SLICESTokenizer::get_vocab,
			 "Get the vocabulary dictionary")
		.def("save", &SLICESTokenizer::save,
			 "Save tokenizer configuration to JSON file",
			 py::arg("filepath"))
		.def("load", &SLICESTokenizer::load,
			 "Load tokenizer configuration from JSON file",
			 py::arg("filepath"))
		.def("print_vocab_stats", &SLICESTokenizer::print_vocab_stats,
			 "Print vocabulary statistics");
	
	// Utility function
	m.def("load_training_data_from_file", &load_training_data_from_file,
		  "Load training data from a text file",
		  py::arg("filepath"));
}
