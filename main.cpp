#include "slices_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <chrono>

#include <fmt/core.h>

int main() {
	try {
		std::cout << "=== SLICES Tokenizer Demo ===" << std::endl;
		
		SLICESTokenizer tokenizer;
		
		std::vector<std::string> training_data = {
			"Ga Bi Bi S S S S Cl 0 3 --o 0 5 oo- 0 6 o-- 0 6 -o- 0 4 -oo 0 4 o-o",
			"Li Na K Rb Cs 1 2 +++ 1 3 --- 2 4 ooo 2 5 +-o 3 6 o+- 4 7 -+o",
			"H He Li Be B C N O F Ne 0 1 ++- 0 2 --+ 1 3 o-+ 2 4 +o- 3 5 -o+",
			"Al Si P S Cl Ar 5 8 ooo 6 9 +-- 7 10 -++ 8 11 o+o 9 12 +-+ 10 13 -o-",
			"Ca Sc Ti V Cr Mn Fe Co Ni Cu 2 7 oo+ 3 8 ++o 4 9 --o 5 10 o-- 6 11 +o+",
			"H H O 0 1 o-o 0 2 o-o 1 2 --o",  // Water-like
			"C C C C 0 1 ooo 1 2 ooo 2 3 ooo 3 0 ooo",  // Cyclic carbon
		};
		
		fmt::println("\n--- Training Tokenizer ---");
		auto start = std::chrono::high_resolution_clock::now();
		
		tokenizer.train_from_iterator(training_data, 1000);
		
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		fmt::println("Training completed in {} ms", duration.count());
		
		tokenizer.print_vocab_stats();
		
		std::string save_path = "slices_tokenizer.json";
		tokenizer.save(save_path);
		
		fmt::println("\n--- Testing Encoding ---");
		std::string test_string = "Ga Bi Bi S S S S Cl 0 3 --o 0 5 oo- 0 6 o-- 0 6 -o- 0 4 -oo 0 4 o-o";
		fmt::println("Original: {}", test_string);
		
		auto encoding = tokenizer.encode(test_string);
		
		fmt::println("Tokens ({}): ['{}']", encoding.tokens.size(), fmt::join(encoding.tokens, "', '"));
		fmt::println("IDs: [{}]", fmt::join(encoding.ids, ", "));
		
		std::string decoded = tokenizer.decode(encoding.ids);
		fmt::println("Decoded: {}", decoded);
		fmt::println("Round-trip successful: {}", (test_string == decoded ? "YES" : "NO"));
		
		fmt::println("\n--- Bond Descriptor Analysis ---");
		int bond_count = 0;
		std::regex bond_pattern(R"([+\-o]{3})");
		for (const auto& token : encoding.tokens) {
			if (std::regex_match(token, bond_pattern)) {
				fmt::println("  Bond descriptor preserved: '{}'", token);
				bond_count++;
			}
		}
		fmt::println("Total bond descriptors found: {}", bond_count);
		
		fmt::println("\n--- Testing Load from File ---");
		SLICESTokenizer loaded_tokenizer;
		loaded_tokenizer.load(save_path);
		
		auto loaded_encoding = loaded_tokenizer.encode(test_string);
		bool same_tokens = (encoding.tokens == loaded_encoding.tokens);
		bool same_ids = (encoding.ids == loaded_encoding.ids);
		
		fmt::println("Loaded tokenizer produces same tokens: {}", (same_tokens ? "YES" : "NO"));
		fmt::println("Loaded tokenizer produces same IDs: {}", (same_ids ? "YES" : "NO"));
		
		fmt::println("\n--- Special Tokens ---");
		fmt::println("[UNK] token ID: {}", loaded_tokenizer.token_to_id("[UNK]"));
		fmt::println("[PAD] token ID: {}", loaded_tokenizer.token_to_id("[PAD]"));
		fmt::println("[CLS] token ID: {}", loaded_tokenizer.token_to_id("[CLS]"));
		fmt::println("[SEP] token ID: {}", loaded_tokenizer.token_to_id("[SEP]"));
		
		fmt::println("\n--- Performance Test ---");
		const int num_iterations = 1000;
		start = std::chrono::high_resolution_clock::now();
		
		for (int i = 0; i < num_iterations; ++i) {
			auto enc = loaded_tokenizer.encode(test_string);
			auto dec = loaded_tokenizer.decode(enc.ids);
		}
		
		end = std::chrono::high_resolution_clock::now();
		auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		double avg_time = static_cast<double>(duration2.count()) / num_iterations;
		
		fmt::println("Average encode+decode time: {:.2f} microseconds", avg_time);
		fmt::println("\n=== Demo Complete ===");
	} catch (const std::exception& e) {
		fmt::print(stderr, "Error: {}\n", e.what());
		return 1;
	}
	
	return 0;
}
