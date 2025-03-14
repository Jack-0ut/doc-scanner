# Mobile Document Scanner

This project implements a mobile document scanner using Python, OpenCV, and computer vision techniques. It allows you to transform images of documents captured at angles into a top-down, "bird's eye view" scan.

## Features

* **Edge Detection:** Identifies the edges of the document.
* **Contour Detection:** Locates the document's outline.
* **Perspective Transform:** Corrects the document's perspective.
* **Adaptive Thresholding:** Converts the scanned image to black and white.

## Prerequisites

* Python 3.x
* pip (Python package installer)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/doc-scanner.git](https://www.google.com/search?q=https://github.com/your-username/doc-scanner.git)
    cd doc-scanner
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To scan a document, run the `scan.py` script with the path to the image:

```bash
python scan.py --image images/your_document.jpg
```