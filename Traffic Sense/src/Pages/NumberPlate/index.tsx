import React, { useRef, useState, type DragEvent } from "react";
import "./style.css";
import { MdOutlineFileUpload } from "react-icons/md";
import { PiImageSquareBold } from "react-icons/pi";
import { LiaFileVideo } from "react-icons/lia";
import { RxCross2 } from "react-icons/rx";
import { LuSparkles } from "react-icons/lu";
import { BiCheckCircle } from "react-icons/bi";
import { runAINumberPlate } from "../../apis/NumberPlate";

const NumberPlate: React.FC = () => {
  const uploadRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [markers, setMarkers] = useState<{ x: number; y: number }[]>([]);
  const [outputPath, setOutputPath] = useState<string | null>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) showPreview(file);
  };

  const handleBrowseClick = () => uploadRef.current?.click();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) showPreview(file);
  };

  const showPreview = (file: File) => {
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setShowResults(false);
    setMarkers([]);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setIsLoading(false);
    setShowResults(false);
    setMarkers([]);
  };

  const handleReRun = () => {
    setShowResults(false);
  };

  const handleNewFile = () => {
    handleRemoveFile();
  };

  const handleRunModel = async () => {
    setIsLoading(true);
    setShowResults(false);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile!);
      const response = await runAINumberPlate(formData, selectedFile!);

      if (response && response.output_path) {
        setOutputPath(response.output_path);
        setShowResults(true);
      }

      setIsLoading(false);
    } catch (error) {
      console.log(error);
      setIsLoading(false);
    }
  };

  return (
    <div className="speed-detection-container">
      <div className="page-title">Number Plate Detection</div>
      <div className="page-subtitle">
        Automatically detect and recognize license plate numbers from images and
        videos. High accuracy AI based technology extracts plate information for
        security and tracking purposes.
      </div>

      {/* File not selected → show upload area */}
      {!selectedFile ? (
        <div className="upload-container">
          <div
            className={`upload-area ${isDragging ? "dragging" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleBrowseClick}
          >
            <div className="upload-icon-container">
              <div className="cloud-icon">
                <MdOutlineFileUpload className="upload-icon" />
              </div>
            </div>
            <div className="upload-text">Drop your file or browse</div>
            <div className="supports-text">
              Supports images (JPG, PNG) and videos (MP4, AVI)
            </div>
            <input
              ref={uploadRef}
              type="file"
              accept="image/jpeg,image/jpg,image/png,video/mp4,video/avi"
              style={{ display: "none" }}
              onChange={handleFileSelect}
            />
            <div className="media-buttons-container ">
              <button className="media-button" title="Upload image here">
                <PiImageSquareBold className="media-icon" /> <span>Images</span>{" "}
              </button>
              <button className="media-button" title="Upload video here">
                <LiaFileVideo className="media-icon" /> <span>Videos</span>{" "}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="content-container">
          <div
            className={`preview-wrapper ${isLoading ? "loading" : ""} ${
              showResults ? "results" : ""
            }`}
          >
            {!isLoading && !showResults && (
              <button className="close-btn" onClick={handleRemoveFile}>
                <RxCross2 className="close-svg" />
              </button>
            )}
            {isLoading ? (
              <div className="loading-wrapper">
                <div className="stars-container">
                  <span className="star">
                    <LuSparkles />
                  </span>
                </div>
                <div className="loading-text">Processing with AI Model...</div>
                <div className="loading-subtext">
                  Analyzing frames and detecting patterns
                </div>
                <div className="dots-container">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            ) : showResults ? (
              <div className="results-wrapper">
                <div className="completion-banner">
                  <BiCheckCircle className="banner-icon" />
                  <div>
                    <div className="banner-title">Plate Detection Complete</div>
                    <div className="banner-subtitle">
                      Inference completed successfully
                    </div>
                  </div>
                </div>
                <div className="results-grid two-column">
                  <div className="annotated-output">
                    <div className="section-title">Processed Output</div>
                    <div className="video-wrapper">
                      {outputPath ? (
                        outputPath.match(/\.(mp4|mov|avi|webm|mkv)$/i) ? (
                          <video
                            src={`${
                              import.meta.env.VITE_BASE_URL
                            }${outputPath}`}
                            controls
                            className="preview-video"
                          />
                        ) : (
                          <img
                            src={`${
                              import.meta.env.VITE_BASE_URL
                            }${outputPath}`}
                            alt="Processed Output"
                            className="preview-video"
                          />
                        )
                      ) : selectedFile?.type?.startsWith("image/") ? (
                        <img
                          src={previewURL!}
                          alt="Preview"
                          className="preview-video"
                        />
                      ) : (
                        <video
                          src={previewURL!}
                          controls
                          className="preview-video"
                        />
                      )}
                    </div>
                  </div>

                  <div className="annotated-output">
                    <div className="section-title">Original Input</div>
                    <div className="video-wrapper">
                      {selectedFile.type.startsWith("image/") ? (
                        <img
                          src={previewURL!}
                          alt="Preview"
                          className="preview-video"
                        />
                      ) : (
                        <video
                          src={previewURL!}
                          controls
                          className="preview-video"
                        />
                      )}
                    </div>
                  </div>
                </div>

                <div className="divider"></div>
                <div className="action-buttons">
                  <button className="re-run-btn" onClick={handleReRun}>
                    Re-run Analysis
                  </button>
                  <button className="new-file-btn" onClick={handleNewFile}>
                    Analyze New File
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="preview-media">
                  <div className="media-container">
                    {selectedFile.type.startsWith("image/") ? (
                      <img src={previewURL!} alt="Preview" />
                    ) : (
                      <video src={previewURL!} controls />
                    )}
                  </div>

                  <div className="file-info">
                    <div className="file-name">{selectedFile.name}</div>
                    <div className="file-details">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB •{" "}
                      {selectedFile.type.startsWith("image/")
                        ? "Image"
                        : "Video"}
                    </div>
                  </div>

                  {/* Reset points button */}
                </div>
                <div className="btn-preview-clear">
                  {markers.length > 0 && (
                    <button
                      className="clear-btn"
                      onClick={() => setMarkers([])}
                    >
                      Clear Points
                    </button>
                  )}
                  <button className="run-btn" onClick={handleRunModel}>
                    <LuSparkles />
                    Run AI Model
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default NumberPlate;
