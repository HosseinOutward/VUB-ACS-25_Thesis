import numpy as np

# todo replace the arithmetic coding with rsna

# todo send a set of bits to show bins being used and only use existing bins
def arithmetic_encode(list_of_ints, symbol_count, out_element_bit_size=64):
    encode_dtype = {'64': np.uint64, '32': np.uint32,
                    '16': np.uint16, '8': np.uint8}[str(out_element_bit_size)]
    # symbol_count = len(np.unique(list_of_ints))
    bit_per_symbol = np.log2(symbol_count)
    symbol_per_out_element = int(out_element_bit_size // bit_per_symbol)
    out_element_count = int((len(list_of_ints) + symbol_per_out_element - 1) // symbol_per_out_element)
    out_res = np.zeros(out_element_count, dtype=encode_dtype)

    for i in range(0, len(list_of_ints), symbol_per_out_element):
        chunk = list_of_ints[i:i + symbol_per_out_element]
        out_idx = i // symbol_per_out_element
        acc = 0
        for j, v in enumerate(chunk):
            assert v < symbol_count, "Index out of bounds for symbol count"
            acc += int(v) * (symbol_count ** j)
        out_res[out_idx] = acc
    return out_res


def arithmetic_decode(encoded_array, symbol_count, orig_size):
    bit_per_symbol = np.log2(symbol_count)
    out_element_bit_size = encoded_array.dtype.itemsize * 8
    symbol_per_out_element = int(out_element_bit_size // bit_per_symbol)

    orig_array = []
    for x in encoded_array:
        temp = int(x)
        for _ in range(symbol_per_out_element):
            orig_array.append(temp % symbol_count)
            temp //= symbol_count
    return np.array(orig_array[:orig_size], dtype=int)


if __name__ == '__main__':
    def test_arithmetic_coding():
        """Test arithmetic coding with various parameter combinations"""
        print("Testing arithmetic coding functions...")

        # Test cases with different configurations
        test_cases = [
            # (array_size, symbol_count, bit_size, description)
            (10, 2, 8, "Small binary array with 8-bit encoding"),
            (20, 4, 16, "Medium quaternary array with 16-bit encoding"),
            (50, 8, 32, "Large octal array with 32-bit encoding"),
            (100, 16, 64, "Very large hexadecimal array with 64-bit encoding"),
            (5, 3, 64, "Small ternary array with 64-bit encoding"),
            (1, 2, 8, "Single element array"),
            (37, 7, 32, "Prime-sized array with 7 symbols"),
            (64, 2, 64, "Power-of-2 sized binary array"),
        ]

        all_tests_passed = True

        for i, (array_size, symbol_count, bit_size, description) in enumerate(test_cases):
            print(f"\nTest {i+1}: {description}")
            print(f"  Array size: {array_size}, Symbol count: {symbol_count}, Bit size: {bit_size}")

            try:
                # Generate random test array
                np.random.seed(42 + i)  # For reproducible results
                test_array = np.random.randint(0, symbol_count, size=array_size)

                # Encode
                encoded = arithmetic_encode(test_array, symbol_count, bit_size)

                # Decode
                decoded = arithmetic_decode(encoded, symbol_count, array_size)

                # Verify correctness
                arrays_equal = np.array_equal(test_array, decoded)

                if arrays_equal:
                    print(f"  ✓ PASSED: Original and decoded arrays match")
                    print(f"  Original size: {len(test_array)}, Encoded size: {len(encoded)} elements")
                    compression_ratio = len(test_array) / len(encoded) if len(encoded) > 0 else float('inf')
                    print(f"  Compression ratio: {compression_ratio:.2f}:1")
                else:
                    print(f"  ✗ FAILED: Arrays do not match")
                    print(f"  Original:  {test_array}")
                    print(f"  Decoded:   {decoded}")
                    all_tests_passed = False

            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                all_tests_passed = False

        # Additional edge case tests
        print("\n" + "="*50)
        print("Testing edge cases...")

        edge_cases = [
            # Test with all zeros
            (np.zeros(10, dtype=int), 2, 32, "All zeros array"),
            # Test with all maximum values
            (np.full(10, 3), 4, 32, "All maximum values"),
            # Test with alternating pattern
            (np.array([0, 1, 0, 1, 0, 1]), 2, 16, "Alternating pattern"),
            # Test with ascending sequence
            (np.array([0, 1, 2, 3, 0, 1, 2, 3]), 4, 32, "Ascending sequence"),
        ]

        for i, (test_array, symbol_count, bit_size, description) in enumerate(edge_cases):
            print(f"\nEdge case {i+1}: {description}")
            try:
                encoded = arithmetic_encode(test_array, symbol_count, bit_size)
                decoded = arithmetic_decode(encoded, symbol_count, len(test_array))

                if np.array_equal(test_array, decoded):
                    print(f"  ✓ PASSED: {description}")
                else:
                    print(f"  ✗ FAILED: {description}")
                    print(f"  Expected: {test_array}")
                    print(f"  Got:      {decoded}")
                    all_tests_passed = False

            except Exception as e:
                print(f"  ✗ ERROR in {description}: {str(e)}")
                all_tests_passed = False

        # Test invalid inputs
        print("\n" + "="*50)
        print("Testing invalid inputs...")

        try:
            # Test with symbol value >= symbol_count (should raise assertion error)
            invalid_array = np.array([0, 1, 2, 5])  # 5 >= symbol_count=4
            arithmetic_encode(invalid_array, 4, 32)
            print("  ✗ FAILED: Should have raised assertion error for invalid symbol")
            all_tests_passed = False
        except AssertionError:
            print("  ✓ PASSED: Correctly caught invalid symbol value")
        except Exception as e:
            print(f"  ✗ UNEXPECTED ERROR: {str(e)}")
            all_tests_passed = False

        # Final result
        print("\n" + "="*50)
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED! Arithmetic coding implementation is correct.")
        else:
            print("❌ SOME TESTS FAILED! Please check the implementation.")

        return all_tests_passed

    # Run the tests
    test_arithmetic_coding()
