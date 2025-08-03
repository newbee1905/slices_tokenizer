#ifndef SLICES_TOKENIZER_HPP
#define SLICES_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <ranges>

#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

using json = nlohmann::json;

struct TokenSpan {
	std::string token;
	size_t start;
	size_t end;
	
	TokenSpan(const std::string& t, size_t s, size_t e) : token(t), start(s), end(e) {}
};

struct Encoding {
	std::vector<std::string> tokens;
	std::vector<int> ids;
	std::vector<TokenSpan> spans;
	
	void clear() {
		tokens.clear();
		ids.clear();
		spans.clear();
	}
};

class SLICESTokenizer {
private:
	std::unordered_map<std::string, int> vocab_;
	std::unordered_map<int, std::string> id_to_token_;
	std::vector<std::string> special_tokens_;
	std::unordered_set<std::string> special_tokens_set_;
	std::regex slices_pattern_;
	std::regex element_pattern_;
	std::regex number_pattern_;
	std::regex bond_pattern_;
	
	// Special token IDs
	int unk_token_id_;
	int pad_token_id_;
	int cls_token_id_;
	int sep_token_id_;
	int mask_token_id_;
	int eos_token_id_;
	
	// Configuration
	int vocab_size_;
	std::string unk_token_;

public:
	SLICESTokenizer() : 
		slices_pattern_(R"([A-Z][a-z]?|\d+|[+\-o]{3})"),
		element_pattern_(R"([A-Z][a-z]?)"),
		number_pattern_(R"(\d+)"),
		bond_pattern_(R"([+\-o]{3})"),
		vocab_size_(1000),
		unk_token_("[UNK]") {
		
		// Initialize special tokens
		special_tokens_ = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"};
		special_tokens_set_ = {special_tokens_.begin(), special_tokens_.end()};
		
		// Initialize special token IDs
		unk_token_id_ = 0;
		pad_token_id_ = 1;
		cls_token_id_ = 2;
		sep_token_id_ = 3;
		mask_token_id_ = 4;
		eos_token_id_ = 5;
	}
	
	// Pre-tokenize a string into SLICES components
	std::vector<TokenSpan> pre_tokenize(const std::string& text) const {
		std::vector<TokenSpan> tokens;
		tokens.reserve(text.length() / 2); 

		for (size_t i = 0; i < text.length(); ) {
			size_t start = i;
			size_t token_len = 0;

			// Case 1: Element ([A-Z][a-z]?)
			if (std::isupper(text[i])) {
				token_len = 1;
				if (i + 1 < text.length() && std::islower(text[i + 1])) {
					token_len = 2;
				}
			}

			// Case 2: Number (\d+)
			else if (std::isdigit(text[i])) {
				token_len = 1;
				while (i + token_len < text.length() && std::isdigit(text[i + token_len])) {
					token_len++;
				}
			}

			// Case 3: Bond descriptor ([+\-o]{3})
			else if (i + 2 < text.length() && 
					 (text[i] == '+' || text[i] == '-' || text[i] == 'o') &&
					 (text[i+1] == '+' || text[i+1] == '-' || text[i+1] == 'o') &&
					 (text[i+2] == '+' || text[i+2] == '-' || text[i+2] == 'o')) 
			{
				token_len = 3;
			}

			if (token_len > 0) {
				tokens.emplace_back(TokenSpan{text.substr(start, token_len), start, start + token_len});
				i += token_len;
			} else {
				// Skip unknown characters or handle them as errors
				++i;
			}
		}
		return tokens;
	}
	
	// Train the tokenizer from an iterator of strings
	void train_from_iterator(const std::vector<std::string>& training_data, int vocab_size = 1000) {
		vocab_size_ = vocab_size;
		vocab_.clear();
		id_to_token_.clear();
		
		for (size_t i = 0; i < special_tokens_.size(); ++i) {
			fmt::println("{}", special_tokens_[i]);
			vocab_[special_tokens_[i]] = static_cast<int>(i);
			id_to_token_[static_cast<int>(i)] = special_tokens_[i];
		}
		
		std::unordered_map<std::string, int> token_counts;
		
		fmt::print("Collecting tokens from training data...\n");
		for (const auto& text : training_data) {
			auto tokens = pre_tokenize(text);
			for (const auto& token_span : tokens) {
				token_counts[token_span.token]++;
			}
		}
		
		std::vector<std::pair<std::string, int>> sorted_tokens(token_counts.begin(), token_counts.end());
		std::sort(sorted_tokens.begin(), sorted_tokens.end(), 
				 [](const auto& a, const auto& b) { return a.second > b.second; });
		
		int token_id = static_cast<int>(special_tokens_set_.size());
		for (const auto& [token, count] : sorted_tokens) {

			if (token_id >= vocab_size_) {
				break;
			}

			if (vocab_.find(token) == vocab_.end()) { 
				vocab_[token] = token_id;
				id_to_token_[token_id] = token;
				token_id++;
			}
		}
		
		fmt::print("Vocabulary built with {} tokens\n", vocab_.size());
		fmt::print("Token frequency examples:\n");
		for (const auto& pair : sorted_tokens | std::views::take(10)) {
			fmt::print("  '{}': {}\n", pair.first, pair.second);
		}
	}
	
	Encoding encode(const std::string& text) {
		Encoding encoding;
		auto token_spans = pre_tokenize(text);
		
		for (const auto& token_span : token_spans) {
			encoding.spans.push_back(token_span);

			auto token = std::string(token_span.token);

			encoding.tokens.push_back(token);
			
			auto it = vocab_.find(token);
			if (it != vocab_.end()) {
				encoding.ids.push_back(it->second);
			} else {
				encoding.ids.push_back(unk_token_id_);
			}
		}
		
		return encoding;
	}
	
	// Decode token IDs back to string
	std::string decode(const std::vector<int>& ids) {
		std::vector<std::string> tokens;
		
		for (int id : ids) {
			auto it = id_to_token_.find(id);

			if (it != id_to_token_.end()) {
				tokens.push_back(it->second);
			} else {
				tokens.push_back(unk_token_);
			}
		}
		
		return fmt::format("{}", fmt::join(tokens, " "));
	}
	
	int token_to_id(const std::string& token) {
		auto it = vocab_.find(token);
		return (it != vocab_.end()) ? it->second : unk_token_id_;
	}
	
	std::string id_to_token(int id) {
		auto it = id_to_token_.find(id);
		return (it != id_to_token_.end()) ? it->second : unk_token_;
	}
	
	int get_vocab_size() const {
		return static_cast<int>(vocab_.size());
	}
	
	const std::unordered_map<std::string, int>& get_vocab() const {
		return vocab_;
	}
	
	void save(const std::string& filepath) {
		json config;
		
		config["model_type"] = "SLICESTokenizer";
		config["vocab_size"] = vocab_size_;
		config["unk_token"] = unk_token_;
		
		config["special_tokens"] = json::object();
		config["special_tokens"]["unk_token"] = unk_token_;
		config["special_tokens"]["pad_token"] = "[PAD]";
		config["special_tokens"]["cls_token"] = "[CLS]";
		config["special_tokens"]["sep_token"] = "[SEP]";
		config["special_tokens"]["mask_token"] = "[MASK]";
		config["special_tokens"]["eos_token"] = "[EOS]";
		
		config["vocab"] = json::object();
		for (const auto& [token, id] : vocab_) {
			config["vocab"][token] = id;
		}
		
		config["pattern"] = R"([A-Z][a-z]?|\d+|[+\-o]{3})";
		
		std::ofstream file(filepath);
		if (file.is_open()) {
			file << config.dump(2);  // Pretty print with 2-space indentation
			file.close();
			fmt::print("Tokenizer saved to {}\n", filepath);
		} else {
			throw std::runtime_error("Could not open file for writing: " + filepath);
		}
	}
	
	// Load tokenizer configuration from JSON file
	void load(const std::string& filepath) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file for reading: " + filepath);
		}
		
		json config;
		file >> config;
		file.close();
		
		vocab_size_ = config["vocab_size"];
		unk_token_ = config["unk_token"];
		
		vocab_.clear();
		id_to_token_.clear();
		
		for (auto& [token, id] : config["vocab"].items()) {
			int token_id = id.get<int>();
			vocab_[token] = token_id;
			id_to_token_[token_id] = token;
		}
		
		if (config.contains("pattern")) {
			std::string pattern = config["pattern"];
			slices_pattern_ = std::regex(pattern);
		}
		
		fmt::print("Tokenizer loaded from {}\n", filepath);
		fmt::print("Vocabulary size: {}\n", vocab_.size());
	}
	
	void print_vocab_stats() {
		fmt::print("\n=== Vocabulary Statistics ===\n");
		fmt::print("Total vocabulary size: {}\n", vocab_.size());
		
		int element_tokens = 0, number_tokens = 0, bond_tokens = 0, special_tokens_count = 0;
		
		for (const auto& [token, id] : vocab_) {
			if (special_tokens_set_.contains(token)) {
				++special_tokens_count;
			} 
			else if (!token.empty() && std::isupper(token[0])) {
				++element_tokens;
			} else if (!token.empty() && std::isdigit(token[0])) {
				++number_tokens;
			} else if (token.length() == 3) {
				++bond_tokens;
			}
		}
		
		fmt::print("  Special tokens: {}\n", special_tokens_count);
		fmt::print("  Element symbols: {}\n", element_tokens);
		fmt::print("  Numbers: {}\n", number_tokens);
		fmt::print("  Bond descriptors: {}\n", bond_tokens);
		
		fmt::print("\nSample bond descriptors in vocabulary:\n");
		int bond_count = 0;
		for (const auto& [token, id] : vocab_) {
			if (std::regex_match(token, bond_pattern_) && bond_count < 10) {
				fmt::print("  '{}' (ID: {})\n", token, id);
				++bond_count;
			}
		}
	}
};

// Helper function to load training data from file
std::vector<std::string> load_training_data_from_file(const std::string& filepath) {
	std::vector<std::string> data;
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open training data file: " + filepath);
	}
	
	std::string line;
	while (std::getline(file, line)) {
		if (!line.empty()) {
			data.push_back(line);
		}
	}
	
	file.close();
	fmt::print("Loaded {} training examples from {}\n", data.size(), filepath);
	return data;
}

#endif // SLICES_TOKENIZER_HPP
