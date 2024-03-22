# FastAPI Image Classification API

This is a FastAPI application for performing image classification using a pre-trained TensorFlow model (VGG19). It exposes an endpoint `/predict/` which accepts image files and returns predictions along with inference time.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Mawue2801/mlcrcscan_api.git
    cd mlcrcscan_api
    ```

2. **Install dependencies:**

    It's recommended to use a virtual environment for managing dependencies.

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model:**

    You need to place your pre-trained model file (`vgg19_bs100_e10.h5`) in the models folder of the project.

4. **Run the FastAPI application:**

    ```bash
    uvicorn main:app --reload
    ```

    The API will start running on `http://127.0.0.1:8000`.

## Usage

1. **Send POST request for prediction:**

    You can send a POST request to the `/predict/` endpoint with an image file to get predictions.

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict/' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@/path/to/your/image.jpg'
    ```

    Replace `/path/to/your/image.jpg` with the path to the image file you want to classify.

2. **Response:**

    Upon successful prediction, you'll receive a JSON response containing the predicted class, prediction probability, and inference time.

    ```json
    {
        "inference_time": 0.129,
        "predicted_class": "colorectal adenocarcinoma epithelium (TUM)",
        "predicted_probability": 0.986
    }
    ```

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests.