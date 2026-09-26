import re
import unicodedata


def normalize_text(text):
    """
    General text normalization for business names and addresses.
    """

    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


if __name__ == "__main__":
    examples = [
        "Pvt. EFS Print Ventures Ltd.",
        "-- Holloway Peak Inc Seafood",
        "Orelee's Barbershop",
        "Door No 183, 41St Cross, 22Nd Main"
    ]

    for example in examples:
        print(f"Original : {example}")
        print(f"Normalized: {normalize_text(example)}")
        print("-" * 50)