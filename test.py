def main():
    # compare between input0.txt and test_all_option0.txt files
    with open("output1.txt", 'r') as f:
        lines = f.readlines()
    with open("test_all_option1.txt", 'r') as f:
        lines2 = f.readlines()
    count = 0
    for i in range(len(lines)):
        # split the line to the input and the output
        inputs, output = lines[i].split()
        inputs2, output2 = lines2[i].split()
        # compare the input
        if output != output2:
            count += 1
    print(f"the number of different lines is: {count}")
    print(f"the precentage of different lines is: {count / len(lines)}")


if __name__ == '__main__':
    main()