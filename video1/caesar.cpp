#include <iostream>
#include <string>


std::string caesar_encrypt(
    std::string s,
    int shift
) {
    std::string encrypted = "";

    for (char& c : s) {
        if (isalpha(c)) {
            char alpha = isupper(c) ? 'A' : 'a';
            c = (c - alpha + shift) % 26 + alpha;
        }

        encrypted += c;
    }

    return encrypted;
}

std::string caesar_decrypt(
    std::string s,
    int shift
) {
    std::string decrypted = "";

    for (char& c : s) {
        if (isalpha(c)) {
            char alpha = isupper(c) ? 'A' : 'a';
            c = (c - alpha - shift + 26) % 26 + alpha;
        }

        decrypted += c;
    }

    return decrypted;
}


int main() {
    std::string text = "Hello, World!";

    int shift = 3;

    std::string encoded = caesar_encrypt(text, shift);
    std::string decrypted = caesar_decrypt(encoded, shift);

    std::cout << "Encoded text: " << encoded << std::endl;
    std::cout << "Decrypted text: " << decrypted << std::endl;

    return 0;
}
