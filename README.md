# VoxPulse Sentinel

Welcome to VoxPulse Sentinel – an advanced and vigilant guardian for your voice assistant interactions. Combining state-of-the-art technology with responsive detection mechanisms, VoxPulse Sentinel stands as a sentinel, ever-watchful for your command.

## Table of Contents
- [Introduction](#introduction)
- [Architecture and Key Features Technology](#architecture-and-key-features-technology)
- [Installation](#installation)
- [Build and Test](#build-and-test)
- [Explore Our Kernel 🚀](#explore-our-kernel-)
- [Technology Stack](#technology-stack)
- [License](#license)
- [Contributing](#contributing)
- [Credits and Acknowledgements](#credits-and-acknowledgements)
- [Contact Information](#contact-information)


## Introduction
At the heart of voice-activated technology lies the magic of hotword detection – the subtle art of identifying specific trigger phrases that awaken voice assistants. VoxPulse Sentinel is our innovative foray into this realm. Engineered with precision, our project's mission is to reliably detect the chosen wake-up phrase, "HeyHoloo." This isn't just about recognition; it's about creating a seamless bridge between user intent and technology responsiveness.

## Architecture and Key Features Technology
At the core of VoxPulse Sentinel lies the [Densent](https://doi.ieeecomputersociety.org/10.1109/CVPR.2017.243) model, originally celebrated in the realm of image processing but now reimagined for audio classification. Our implementation is a testament to its adaptability and strength in new domains, as highlighted in the insightful study, [Rethinking CNN Models for Audio Classification](https://www.researchgate.net/publication/343150435_Rethinking_CNN_Models_for_Audio_Classification).

Our model's structure is meticulously fine-tuned across three distinct data classes:

- **'Hey Holoo' Class**: The centerpiece of our system, this class is intricately trained to respond exclusively to the "HeyHoloo" hotword.
- **Noice Class**: Incorporates phrases such as "Hey Siri" and "Hey Google",etc. enabling our model to distinguish between closely related audio cues. These phrases helped the model to distinguish other phrases from "HeyHoloo".
- **'Others' Class**: A broad and diverse category encompassing all other audio, ensuring our system's comprehensive understanding and responsiveness.

---

Designed to excel in real-time scenarios, VoxPulse Sentinel demonstrates remarkable efficiency in distinguishing the 'HeyHoloo' phrase amidst a myriad of soundscapes. This balance of precision and versatility sets our system apart in the field of audio classification. The model also is designed to act as ***Command Detection*** as well. It can tell if a command is said in an audio file and when the command has happened. For now, only the 'HeyHoloo' is recognized. It is worth mentioning that anything else other than 'HeyHoloo' will not be recognized by our system.


## Installation
Follow these steps to set up VoxPulse Sentinel on your system:

1. **Download the Pretrained Model Weights**:
   - First, download the [pretrained model weight](https://drive.google.com/file/d/13R9FrXqcnUg55y4sjKaTAlYD5Q77NBit/view?usp=sharing).
   - Place the downloaded file into the ***models*** directory within your VoxPulse Sentinel project folder: .

2. **Install Required Libraries**:
   - Open your terminal or command prompt.
   - Navigate to the root directory of the VoxPulse Sentinel project.
   - Run the following commands to install necessary libraries:

```bash
sudo apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
pip install -r requirements.txt
```


## Build and Test

### Setting Up Your Environment
Begin by preparing your audio data, ensuring it's ready for use with the pre-trained model. Training the model from scratch is optional since the model has already been trained.

#### Data Preparation
Uniform audio length is critical. We standardize all audio data to 3 seconds.

- **Padding Short Audio**: For 'heyholoo' and similar classes.

  ```bash
  cd src/densenet/data
  python data_pad.py --input_dir [input data dir] --output_dir [output data dir] --len 3 --sr [desired sample rate]
  ```

- **Processing 'Others' Class**: Using the [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets).

  ```bash
  cd src/densenet/data
  python3 data_split.py --input_dir [input data dir] --output_dir [output data dir] --len 3 --sr [desired sample rate]
  ```

#### Data Augmentation
Enhance your dataset's robustness and variety.

- **Pitch Shift Augmentation**

  ```bash
  cd src/densenet/data
  python3 augment.py --input_dir [input data dir] --output_dir [output data dir] --step [pitchshift step] --sr [desired sample rate]
  ```

- **Adding Noise**

  ```bash
  cd src/densenet/data
  python3 add_noise.py --input_dir [input data dir] --noise_dir [noise files dir] --output_dir [output data dir] --sr [desired sample rate]
  ```


#### Feature Extraction
Prepare your dataset for potential retraining or validation.

```bash
cd src/densenet/features
python preprocess.py --dataset [dataset main directory] --train_np [train spectrograms dir] --valid_np [validation spectrograms dir] --train_csv [path to train.csv] --valid_csv [path to valid.csv]
```

### Model Training (Optional)
The model comes pre-trained. Retraining is optional and can be done with your prepared data.

```bash
cd src/densenet/train
python train.py --checkpoint_dir [checkpoint save dir] --logdir [logs save dir] --train_csv [path to train.csv] --valid_csv [path to valid.csv] --name [model name]
```

### Model Evaluation
Evaluate the model’s performance and accuracy.
```bash
cd src/densenet/evaluation
python eval.py --input_dir [input audio files dir] --temp_dir [temp dir] --sr [desired sample rate]
```

### Testing the Model
Test the pre-trained model in various scenarios to assess its real-world application.

#### Offline Test
Test with prerecorded audio containing the key phrase.
```bash
cd src/densenet/test
python offline_test.py --input [input audio file] --chime [chime file path] --output_dir [output dir] --sr [desired sample rate]
```

#### Online Test
Test the model's responsiveness in real-time using a microphone.
```bash
cd src/densenet/test
python online_test.py
```

# Explore Our Kernel 🚀
We are thrilled to unveil our cutting-edge kernel, an embodiment of innovation that integrates the audio manipulation capabilities of VoxArte Studio! It's not just a repository; it's a revolution in audio processing, built with our audio projects at its heart.

## Catch the Wave of Audio Innovation
Don't miss out on this opportunity to be a part of the audio evolution. Click the link blow, star the repo for future updates, and let your ears be the judge. If you're as passionate about audio as we are, we look forward to seeing you there!

Remember, the future of audio is not just heard; it's shared and shaped by enthusiasts and professionals alike. Let's make waves together with VoxArte Studio and our Kernel. 🚀

🔗 [Kernel Repository](https://github.com/Meta-Intelligence-Services)

---

For any queries or discussions regarding our kernel, feel free to open an issue in the kernel's repository, and we'll be more than happy to engage with you. Together, we're not just changing audio; we're making history!

## Technology Stack
VoxArte Studio harnesses a collection of powerful libraries and frameworks to provide its audio processing capabilities:

Great, thanks for providing the list of technologies used in VoxPulseSentinel. I'll add a revised "Technology Stack" section to the README which outlines these libraries and frameworks:

---

## Technology Stack
VoxPulseSentinel is built on a foundation of powerful libraries and frameworks, each contributing to its robust functionality:

- **TensorFlow (2.13.0)**: A comprehensive open-source platform for machine learning, used for building and training the Densent model.
- **PyAudio (0.2.13)**: Provides Python bindings for PortAudio, the cross-platform audio I/O library, crucial for audio capture and playback.
- **NumPy (1.24.3)**: Essential for high-level mathematical functions and multi-dimensional arrays manipulation.
- **FFmpeg (1.4)**: A complete solution for recording, converting, and streaming audio and video.
- **Pydub (0.25.1)**: A simple and easy-to-use Python module to work with audio data.
- **Soundfile (0.12.1)**: Enables reading and writing of audio files in various formats.
- **Librosa (0.10.1)**: A Python package for music and audio analysis, providing tools to analyze and extract audio features.
- **Pandas (2.1.1)**: Used for data manipulation and analysis, particularly useful for handling large data sets.
- **Scikit-learn (1.3.1)**: Implements a wide range of machine learning algorithms for medium-scale supervised and unsupervised problems.
- **Pedalboard**: A Python library for adding effects to audio.
- **Wave**: A standard Python module for working with WAV files.
- **TQDM**: Offers a fast, extensible progress bar for loops and other iterative processing.

---

Make sure to install all required libraries using the `requirements.txt` file - which exists in each project's directory - or manually install them with `pip` if necessary.

## License
VoxPulse Sentinel is open-sourced under the MIT License. See [LICENSE](LICENSE) for more details.

## Contributing
While we deeply value community input and interest in VoxPulse Sentinel, the project is currently in a phase where we're mapping out our next steps and are not accepting contributions just yet. We are incredibly grateful for your support and understanding. Please stay tuned for future updates when we'll be ready to welcome contributions with open arms.

## Credits and Acknowledgements
We would like to extend our heartfelt thanks to Mrs.Arefe Khaleghi for her guidance and wisdom throughout the development of VoxPulse Sentinel. Her insights have been a beacon of inspiration for this project.

## Contact Information
Although we're not open to contributions at the moment, your feedback and support are always welcome. Please feel free to star the project or share your thoughts through the Issues tab on GitHub, and we promise to consider them carefully.please [open an issue](https://github.com/Amir-Nassimi/VoxPulse-Sentinel/issues) in the VoxPulse Sentinel repository, and we will assist you.