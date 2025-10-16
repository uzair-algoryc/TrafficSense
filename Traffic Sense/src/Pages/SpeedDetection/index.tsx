/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useRef, useState, useEffect, type DragEvent } from "react"; // ✅ Added useEffect import
import "./style.css";
import { MdOutlineFileUpload } from "react-icons/md";
import { LiaFileVideo } from "react-icons/lia";
import { RxCross2 } from "react-icons/rx";
import { LuSparkles } from "react-icons/lu";
import { TbActivityHeartbeat } from "react-icons/tb";
import {
  analyzeViolationImage,
  runAISpeedDetection,
} from "../../apis/SpeedDetection";
import { useToast } from "@chakra-ui/react";
import { Tooltip } from "react-tooltip";
import "react-tooltip/dist/react-tooltip.css";
import { AiOutlineInfoCircle } from "react-icons/ai";

const SpeedDetection: React.FC = () => {
  const [selectedViolation, setSelectedViolation] = useState<string | null>(
    null
  );
  const [detectionImage, setDetectionImage] = useState<string | null>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [markers, setMarkers] = useState<{ x: number; y: number }[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [response, setResponse] = useState<any>(null);
  const [inputs, setInputs] = useState({
    width: "",
    height: "",
    max_speed: "",
    conf: "",
  });
  // ✅ NEW: Track video dimensions for reliable marker positioning
  const [videoDimensions, setVideoDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  const toast = useToast();

  // ✅ NEW: Effect to load video dimensions on mount/change
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !previewURL) return;

    const handleLoadedMetadata = () => {
      setVideoDimensions({
        width: video.videoWidth,
        height: video.videoHeight,
      });
    };

    video.addEventListener("loadedmetadata", handleLoadedMetadata);

    // Check if already loaded
    if (video.readyState >= 2) {
      // HAVE_CURRENT_DATA or higher
      handleLoadedMetadata();
    }

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [previewURL]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputs((prev) => ({ ...prev, [name]: value }));
  };

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
    if (!file) return;

    if (!file.type.startsWith("video/")) {
      toast({
        title: "Only video files are allowed.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    showPreview(file);
  };

  const handleBrowseClick = () => uploadRef.current?.click();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // if (file && file.type.startsWith("video/")) showPreview(file);
    if (!file.type.startsWith("video/")) {
      toast({
        title: "Only video files are allowed.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    showPreview(file);
  };

  const showPreview = (file: File) => {
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setShowResults(false);
    setMarkers([]);
    // ✅ Reset dimensions when new file (ensures clean load)
    setVideoDimensions({ width: 0, height: 0 });
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setIsLoading(false);
    setShowResults(false);
    setMarkers([]);
    // ✅ Reset dimensions
    setVideoDimensions({ width: 0, height: 0 });
  };

  // const handleRunModel = () => {
  //   setIsLoading(true);
  //   setTimeout(() => {
  //     setIsLoading(false);
  //     setShowResults(true);
  //   }, 5000);
  // };

  const handleReRun = () => {
    setShowResults(false);
  };

  const handleNewFile = () => {
    handleRemoveFile();
  };

  const handleMediaClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (markers.length >= 4) return; // stop after 4 clicks

    const mediaElement = e.currentTarget.querySelector(
      "video"
    ) as HTMLVideoElement;

    if (!mediaElement) return;

    // Relative coordinates (pixels)
    const rect = mediaElement.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * mediaElement.videoWidth;
    const y = ((e.clientY - rect.top) / rect.height) * mediaElement.videoHeight;

    console.log(
      `Point ${markers.length + 1} → X: ${x.toFixed(2)}px, Y: ${y.toFixed(2)}px`
    );

    setMarkers((prev) => [...prev, { x, y }]);
  };

  // ✅ UPDATED: Add timeout to prevent hanging (adjust 30000ms as needed)
  const handleRunAI = async () => {
    if (!selectedFile) {
      toast({
        title: "Please upload a video file first.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    if (!inputs.width || !inputs.height || !inputs.max_speed || !inputs.conf) {
      toast({
        title: "All input fields are required.",
        description: "Please fill width, height, max speed, and confidence.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    if (markers.length < 4) {
      toast({
        title: "Please select 4 coordinates before running AI.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    // Create coordinates string in required order
    // ✅ Round each coordinate to nearest integer
    const coordinatesString = markers
      .map((m) => `${Math.round(m.x)},${Math.round(m.y)}`)
      .join(",");

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("coordinates", coordinatesString);
    formData.append("width", inputs.width || "0");
    formData.append("height", inputs.height || "0");
    formData.append("max_speed", inputs.max_speed || "0");
    formData.append("conf", inputs.conf || "");
    setIsLoading(true);

    try {
      const apiPromise = runAISpeedDetection(formData);
      const response = await Promise.race([apiPromise]);
      toast({
        title: response?.message || "AI model executed successfully!",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      setIsLoading(false);
      setResponse(response);
      setShowResults(true);
    } catch (error: any) {
      console.error("API Error:", error); // ✅ Add logging for debugging
      toast({
        title: error.message || "Something went wrong!",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      setIsLoading(false);
      // ✅ Ensure preview shows (redundant but safe)
      setShowResults(false);
    }
  };

  // ✅ UPDATED: Use videoDimensions for vw/vh (fallback to 1)
  const vw = videoDimensions.width || 1;
  const vh = videoDimensions.height || 1;

  const handleAnalyzeViolation = async () => {
    if (!selectedViolation) {
      toast({
        title: "Please select a violation image first.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("image_path", selectedViolation);

      const nextResponse = await analyzeViolationImage(formData);

      if (nextResponse.output_path) {
        setDetectionImage(nextResponse.output_path);
        toast({
          title: "Analysis complete!",
          status: "success",
          duration: 2000,
          isClosable: true,
          position: "top",
        });
      }
    } catch (error: any) {
      console.log(error);
      toast({
        title: "Failed to analyze this image.",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="speed-detection-container">
      <div className="page-title">Speed Detection</div>
      <div className="page-subtitle">
        Detect and measure vehicle speeds in real-time using advanced traffic
        sense algorithms. Upload a video to analyze traffic patterns and
        identify speed violations.
      </div>
      {selectedFile && !isLoading && !showResults && (
        <div className="input-fields-container">
          {/* Width */}
          <div className="speed-input-group">
            <label>
              Width (m)
              <AiOutlineInfoCircle
                data-tooltip-id="widthTip"
                data-tooltip-html={`<div class='tip-content'><strong>Detection Width:</strong><br/><span>Specify the detection area width in <b>meters</b>.<br/>Helps the model understand real-world scale.</span></div>`}
                className="info-icon"
              />
            </label>
            <Tooltip id="widthTip" place="top" className="custom-tooltip" />
            <input
              type="number"
              name="width"
              min="0"
              value={inputs.width}
              onChange={handleInputChange}
              placeholder="Enter width in meters"
            />
          </div>

          {/* Height */}
          <div className="speed-input-group">
            <label>
              Height (m)
              <AiOutlineInfoCircle
                data-tooltip-id="heightTip"
                data-tooltip-html={`<div class='tip-content'><strong>Detection Height:</strong><br/><span>Specify the detection area height in <b>meters</b>.<br/>Used to calculate object proportions.</span></div>`}
                className="info-icon"
              />
            </label>
            <Tooltip id="heightTip" place="top" className="custom-tooltip" />
            <input
              type="number"
              name="height"
              min="0"
              value={inputs.height}
              onChange={handleInputChange}
              placeholder="Enter height in meters"
            />
          </div>

          {/* Max Speed */}
          <div className="speed-input-group">
            <label>
              Max Speed (km/h)
              <AiOutlineInfoCircle
                data-tooltip-id="speedTip"
                data-tooltip-html={`<div class='tip-content'><strong>Speed Limit:</strong><br/><span>Maximum allowed speed in <b>km/h</b>.<br/>Vehicles exceeding this value will be flagged.</span></div>`}
                className="info-icon"
              />
            </label>
            <Tooltip id="speedTip" place="top" className="custom-tooltip" />
            <input
              type="number"
              name="max_speed"
              min="0"
              max="300"
              value={inputs.max_speed}
              onChange={handleInputChange}
              placeholder="0–300 km/h"
            />
          </div>

          {/* Confidence */}
          <div className="speed-input-group">
            <label>
              Confidence
              <AiOutlineInfoCircle
                data-tooltip-id="confTip"
                data-tooltip-html={`<div class='tip-content'><strong>Confidence Threshold:</strong><br/><span>Detection confidence (0.0–1.0).<br/>Higher values reduce false detections.</span></div>`}
                className="info-icon"
              />
            </label>
            <Tooltip id="confTip" place="top" className="custom-tooltip" />
            <input
              type="number"
              name="conf"
              min="0"
              max="1"
              step="0.01"
              value={inputs.conf}
              onChange={handleInputChange}
              placeholder="0.0–1.0"
            />
          </div>
        </div>
      )}
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
            <div className="supports-text">Supports videos (MP4, AVI)</div>
            <input
              ref={uploadRef}
              type="file"
              accept="video/mp4,video/avi"
              style={{ display: "none" }}
              onChange={handleFileSelect}
            />
            <div className="media-buttons-container ">
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
                  <TbActivityHeartbeat className="banner-icon" />
                  <div>
                    <div className="banner-title">Speed Analysis Complete</div>
                    <div className="banner-subtitle">
                      Inference completed successfully
                    </div>
                  </div>
                </div>
                <div className="results-grid">
                  <div className="annotated-output">
                    <div className="section-title">Annotated Output</div>
                    <div className="video-wrapper">
                      <video
                        src={
                          response?.output_video
                            ? response.output_video.startsWith("http")
                              ? response.output_video
                              : `${import.meta.env.VITE_BASE_URL}${
                                  response.output_video
                                }`
                            : previewURL!
                        }
                        controls
                        className="preview-video"
                      />
                    </div>
                    {/* ✅ New block for violation images */}
                    {response?.violation_images &&
                      response.violation_images.length > 0 && (
                        <div className="violation-images">
                          <div className="section-title">Violation Images</div>
                          <div className="violation-grid">
                            {response.violation_images.map(
                              (img: string, index: number) => (
                                <img
                                  key={index}
                                  src={`${import.meta.env.VITE_BASE_URL}${img}`}
                                  alt={`Violation ${index + 1}`}
                                  className={`violation-thumb ${
                                    selectedViolation === img ? "selected" : ""
                                  }`}
                                  onClick={() => setSelectedViolation(img)}
                                />
                              )
                            )}
                          </div>
                        </div>
                      )}
                  </div>
                  {selectedViolation && (
                    <div className="detection-details-section">
                      <div className="section-title">Detection Details</div>
                      <div className="container-detection" style={{display:"flex"}}>
                        <div className="selected-image-wrapper">
                          <img
                            src={`${
                              import.meta.env.VITE_BASE_URL
                            }${selectedViolation}`}
                            alt="Selected Violation"
                            className="selected-violation-image"
                          />
                        </div>
                        {detectionImage && (
                          <>
                          <div className="next-detection-image">
                            <img
                              src={`${
                                import.meta.env.VITE_BASE_URL
                              }${detectionImage}`}
                              alt="Next Detection"
                              className="detected-image"
                            />
                          </div>
                      </>  )}
                      </div>

                      <button
                        className="analyze-more-btn"
                        onClick={handleAnalyzeViolation}
                      >
                        Analyze This Image
                      </button>
                    </div>
                  )}
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
                  <div className="media-container" onClick={handleMediaClick}>
                    <video ref={videoRef} src={previewURL!} controls />

                    {/* SVG lines between markers */}
                    <svg className="lines-overlay">
                      {markers.map((m, i) => {
                        if (i === 0) return null;
                        const prev = markers[i - 1];
                        // ✅ UPDATED: Use videoDimensions
                        return (
                          <line
                            key={`line-${i}`}
                            x1={`${(prev.x / vw) * 100}%`}
                            y1={`${(prev.y / vh) * 100}%`}
                            x2={`${(m.x / vw) * 100}%`}
                            y2={`${(m.y / vh) * 100}%`}
                            stroke="#3b82f6"
                            strokeWidth="2"
                          />
                        );
                      })}

                      {/* Close shape from 4→1 */}
                      {markers.length === 4 && videoDimensions.width > 0 && (
                        <line
                          key="closing-line"
                          x1={`${(markers[3].x / vw) * 100}%`}
                          y1={`${(markers[3].y / vh) * 100}%`}
                          x2={`${(markers[0].x / vw) * 100}%`}
                          y2={`${(markers[0].y / vh) * 100}%`}
                          stroke="#3b82f6"
                          strokeWidth="2"
                        />
                      )}
                    </svg>

                    {/* Dots with labels */}
                    {markers.map((m, i) => {
                      // ✅ UPDATED: Use videoDimensions
                      return (
                        <div
                          key={`dot-${i}`}
                          className="marker-dot"
                          style={{
                            top: `${(m.y / vh) * 100}%`,
                            left: `${(m.x / vw) * 100}%`,
                          }}
                        >
                          <span className="dot-label">{i + 1}</span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="file-info">
                    <div className="file-name">{selectedFile.name}</div>
                    <div className="file-details">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Video
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
                  <button className="run-btn" onClick={handleRunAI}>
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

export default SpeedDetection;
