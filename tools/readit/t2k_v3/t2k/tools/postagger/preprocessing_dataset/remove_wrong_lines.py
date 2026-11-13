
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("-o", "--output_file", default="output_remove_wrong_lines.conllu",
                        help="Output file path")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    with open(output_file, "w") as output:
        with open(input_file, "r") as f:
            for line in f:
                if "\t_@_" in line:
                    print(f"removed {line}")
                    continue
                else:
                    output.write(line)