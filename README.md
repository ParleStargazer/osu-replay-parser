# **osu\!mania Replay Data Analyzer**

This project is a Python script designed to parse and analyze osu\!mania replay files (.osr). It extracts real-time keypress data, calculates the duration of each note, and generates high-resolution statistical distribution charts to help VSRG (Vertical Scrolling Rhythm Game) players analyze their keypress habits and accuracy performance.
~~100% gemini vibe coding ™~~

## **Features**

* **Replay Parsing**: Automatically parses the binary header information of .osr files and the LZMA-compressed action sequence data.  
* **Data Statistics**: Accurately calculates the duration from key press to release based on a state machine logic, supporting parallel multi-key processing.  
* **Visualization**: Utilizes Matplotlib to generate keypress duration distribution charts with 1ms precision scale support.  
* **High-Resolution Output**: Offers multiple resolution presets from HD (720P) to 4K (2160P) to ensure high-quality image output.

## **Requirements**

Running this project requires a Python 3.x environment and the following third-party library:

* matplotlib: For chart plotting and rendering.  
* Built-in modules: struct, lzma, os, sys, collections, etc.

**Install Dependencies:**

pip install matplotlib

## **Usage**

1. **Prepare Replay File**: Ensure you have a valid .osr replay file.  
2. **Launch Script**:  
   python osu\_replay\_parser.py

3. **Input Path**: Follow the prompts to drag the .osr file into the terminal or manually input the full path.  
4. **Select Resolution**: Choose an output resolution from levels 1-4 (Default: 1080P).  
5. **View Results**: The script will generate an analysis chart named \[filename\]\_analysis.png in the same directory.

## **Optimizations & Improvements**

This project is inspired by [adgjl7777777/VSRG\_Total\_Analyzer](https://github.com/adgjl7777777/VSRG_Total_Analyzer) and includes the following optimizations:

1. **Visual Presentation**:  
   * Introduced MultipleLocator for major and minor axis ticks, making the observation of X-axis (intervals) and Y-axis (counts) more intuitive.  
   * Optimized plotting parameters, including dynamic line width scaling based on resolution, adaptive legend positioning, and refined grid line transparency.  
2. **Interaction Experience**: Simplified the command-line interface workflow and supported various resolution export options.  
3. **Code Standards**: Adopted a more rigorous modular structure with detailed professional comments to facilitate secondary development.

## **License**

This project is licensed under the MIT License.
