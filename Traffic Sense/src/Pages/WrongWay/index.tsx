/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useRef, useState, useEffect, type DragEvent } from "react";
import "./style.css";
import { MdOutlineFileUpload } from "react-icons/md";
import { LiaFileVideo } from "react-icons/lia";
import { RxCross2 } from "react-icons/rx";
import { LuSparkles } from "react-icons/lu";
import { RiErrorWarningLine } from "react-icons/ri";
import { runAIWrongway } from "../../apis/WrongWay";
import { useToast } from "@chakra-ui/react";
import { analyzeViolationImage } from "../../apis/SpeedDetection";

const WrongWay: React.FC = () => {
  const [selectedViolation, setSelectedViolation] = useState<string | null>(
    null
  );
  const [detectionImage, setDetectionImage] = useState<string | null>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [markers, setMarkers] = useState<{ x: number; y: number }[]>([]);
  const [oneWay, setOneWay] = useState(true);
  const [twoWay, setTwoWay] = useState(false);
  const [resultsData, setResultsData] = useState<{
    wrong_way_count?: number;
    wrong_way_images?: string[];
    output_video?: string;
  } | null>(null);
  // NEW: Persist dimensions
  const [videoDimensions, setVideoDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  const [imgDimensions, setImgDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  const toast = useToast();

  // NEW: Effect for video metadata
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

    if (video.readyState >= 2) {
      handleLoadedMetadata();
    }

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [previewURL]);

  // NEW: Effect for image load
  useEffect(() => {
    const img = imageRef.current;
    if (!img || !previewURL) return;

    const handleLoad = () => {
      setImgDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };

    img.addEventListener("load", handleLoad);
    if (img.complete) {
      handleLoad();
    }

    return () => {
      img.removeEventListener("load", handleLoad);
    };
  }, [previewURL]);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleRunModel = async () => {
    if (markers.length !== 4 && markers.length !== 8) {
      toast({
        title: "Please mark 4 points per lane before running the model.",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    setIsLoading(true);
    setShowResults(false);

    try {
      const coordinatesString = markers
        .map((m) => `${Math.round(m.x)},${Math.round(m.y)}`)
        .join(",");
      const direction = oneWay ? "up" : "down";

      const formData = new FormData();
      formData.append("file", selectedFile!);
      formData.append("coordinates", coordinatesString);
      formData.append("direction", direction);

      const apiPromise = runAIWrongway(formData);
      const response = await Promise.race([apiPromise]);

      toast({
        title: "Analysis completed successfully!",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top",
      });

      setResultsData(response);
      setShowResults(true);
    } catch (error: any) {
      console.error("Error running model:", error);
      toast({
        title: error.message || "An unexpected error occurred.",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (
      file &&
      (file.type.startsWith("video/") || file.type.startsWith("image/"))
    )
      showPreview(file);
  };

  const handleBrowseClick = () => uploadRef.current?.click();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (
      file &&
      (file.type.startsWith("video/") || file.type.startsWith("image/"))
    )
      showPreview(file);
  };

  const showPreview = (file: File) => {
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setShowResults(false);
    setMarkers([]);
    setVideoDimensions({ width: 0, height: 0 }); // NEW: Reset
    setImgDimensions({ width: 0, height: 0 }); // NEW: Reset
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setIsLoading(false);
    setShowResults(false);
    setMarkers([]);
    setVideoDimensions({ width: 0, height: 0 }); // NEW: Reset
    setImgDimensions({ width: 0, height: 0 }); // NEW: Reset
  };

  const handleReRun = () => {
    setShowResults(false);
  };

  const handleNewFile = () => {
    handleRemoveFile();
  };

  // UPDATED: Use persisted dimensions
  const getMediaDimensions = () => {
    const isImage = selectedFile?.type.startsWith("image/") || false;
    const dims = isImage ? imgDimensions : videoDimensions;
    return {
      width: dims.width || 1,
      height: dims.height || 1,
    };
  };

  const handleMediaClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const numLanes = (oneWay ? 1 : 0) + (twoWay ? 1 : 0);
    if (numLanes === 0) return;

    const maxPoints = numLanes * 4;
    if (markers.length >= maxPoints) return; // stop after max points

    const isImage = selectedFile?.type.startsWith("image/") || false;
    const mediaElement = isImage
      ? (e.currentTarget.querySelector("img") as HTMLImageElement)
      : (e.currentTarget.querySelector("video") as HTMLVideoElement);

    if (!mediaElement) return;

    const rect = mediaElement.getBoundingClientRect();
    const dims = getMediaDimensions();
    if (dims.width <= 1) return; // Wait for load

    // Relative coordinates (pixels)
    const x = ((e.clientX - rect.left) / rect.width) * dims.width;
    const y = ((e.clientY - rect.top) / rect.height) * dims.height;

    console.log(
      `Point ${markers.length + 1} → X: ${x.toFixed(2)}px, Y: ${y.toFixed(2)}px`
    );

    setMarkers((prev) => [...prev, { x, y }]);
  };

  const dims = getMediaDimensions();
  const vw = dims.width;
  const vh = dims.height;
  const isImage = selectedFile?.type.startsWith("image/") || false;

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
  
        if (nextResponse?.output_path) {
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
      <div className="page-title">Wrong-Way Detection</div>
      <div className="page-subtitle">
        Identify vehicles traveling in the wrong direction on roads, parking
        lots, or restricted areas. Real-time alerts help prevent accidents and
        security breaches.
      </div>
      {selectedFile && (
        <div className="lane-config">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={oneWay}
              onChange={(e) => setOneWay(e.target.checked)}
            />
            <span>Up</span>
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={twoWay}
              onChange={(e) => setTwoWay(e.target.checked)}
            />
            <span>Down</span>
          </label>
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
              accept="image/jpeg,image/png,video/mp4,video/avi" // UPDATED: Added images
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
                  <RiErrorWarningLine className="banner-icon" />
                  <div>
                    <div className="banner-title">
                      Wrong-Way Detection Complete
                    </div>
                    <div className="banner-subtitle">
                      Inference completed successfully
                    </div>
                  </div>
                </div>
                <div className="results-grid">
                  <div className="annotated-output">
                    <div className="section-title">Annotated Output</div>
                    <div className="video-wrapper">
                      {isImage ? (
                        <img
                          src={resultsData?.output_video || previewURL!}
                          alt="Preview"
                          className="preview-video"
                        />
                      ) : (
                        <video
                          src={
                            resultsData?.output_video
                              ? `${import.meta.env.VITE_BASE_URL}${
                                  resultsData.output_video
                                }`
                              : previewURL!
                          }
                          controls
                          className="preview-video"
                        />
                      )}
                      {resultsData?.wrong_way_images &&
                        resultsData.wrong_way_images.length > 0 && (
                          <div className="wrong-way-images">
                            {resultsData.wrong_way_images.map((img, index) => (
                              <img
                                key={index}
                                src={`${import.meta.env.VITE_BASE_URL}${img}`}
                                alt={`Wrong way ${index + 1}`}
                                className={`wrong-way-thumbnail ${
                                  selectedViolation === img ? "selected" : ""
                                }`}
                                onClick={() => {
                                  setSelectedViolation(img);
                                  setDetectionImage(null); // reset old detection result
                                }}
                              />
                            ))}
                          </div>
                        )}

                      {/* {mockData.vehicles.map((vehicle) => (
                        <div
                          key={vehicle.id}
                          className="bbox"
                          style={{
                            top: vehicle.bbox.top,
                            left: vehicle.bbox.left,
                            width: vehicle.bbox.width,
                            height: vehicle.bbox.height,
                          }}
                        >
                          <div className="bbox-label">
                            Vehicle #{vehicle.id} {vehicle.confidence}%
                          </div>
                        </div>
                      ))} */}
                    </div>
                  </div>
                  <div className="summary-metrics">
                    <div className="section-title">Summary Metrics</div>
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <div className="metric-label">Wrong-Way Count</div>
                        <div className="metric-value">
                          {resultsData?.wrong_way_count ?? 0}
                        </div>
                      </div>
                      
                    </div>
                    {selectedViolation && (
                    <div className="detection-details">
                      <div className="section-title">Detection Details</div>
                      <img
                        src={`${
                          import.meta.env.VITE_BASE_URL
                        }${selectedViolation}`}
                        alt="Selected Violation"
                        className="detection-preview"
                      />

                      <button
                        className="analyze-more-btn"
                        onClick={handleAnalyzeViolation}
                      >
                        Analyze This Image
                      </button>

                      {detectionImage && (
                        <img
                          src={`${
                            import.meta.env.VITE_BASE_URL
                          }${detectionImage}`}
                          alt="Detection Result"
                          className="detection-result"
                        />
                      )}
                    </div>
                  )}
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
                  <div className="media-container" onClick={handleMediaClick}>
                    {isImage ? (
                      <img ref={imageRef} src={previewURL!} alt="Preview" />
                    ) : (
                      <video ref={videoRef} src={previewURL!} controls />
                    )}

                    {/* SVG lines between markers */}
                    <svg className="lines-overlay">
                      {markers.map((m, i) => {
                        if (i === 0 || i % 4 === 0) return null; // 🔥 Prevent connecting box N to box N+1
                        const prev = markers[i - 1];
                        return (
                          <line
                            key={`line-${i}`}
                            x1={`${(prev.x / vw) * 100}%`} // UPDATED: Use vw/vh
                            y1={`${(prev.y / vh) * 100}%`}
                            x2={`${(m.x / vw) * 100}%`}
                            y2={`${(m.y / vh) * 100}%`}
                            stroke="#3b82f6"
                            strokeWidth="2"
                          />
                        );
                      })}

                      {/* Close first shape (points 0-3) */}
                      {markers.length >= 4 && vw > 0 && (
                        <line
                          key="closing-one"
                          x1={`${(markers[3].x / vw) * 100}%`} // UPDATED: vw/vh
                          y1={`${(markers[3].y / vh) * 100}%`}
                          x2={`${(markers[0].x / vw) * 100}%`}
                          y2={`${(markers[0].y / vh) * 100}%`}
                          stroke="#3b82f6"
                          strokeWidth="2"
                        />
                      )}

                      {/* Close second shape (points 4-7) */}
                      {markers.length >= 8 && vw > 0 && (
                        <line
                          key="closing-two"
                          x1={`${(markers[7].x / vw) * 100}%`} // UPDATED: vw/vh
                          y1={`${(markers[7].y / vh) * 100}%`}
                          x2={`${(markers[4].x / vw) * 100}%`}
                          y2={`${(markers[4].y / vh) * 100}%`}
                          stroke="#3b82f6"
                          strokeWidth="2"
                        />
                      )}
                    </svg>

                    {/* Dots with labels */}
                    {markers.map((m, i) => (
                      <div
                        key={`dot-${i}`}
                        className="marker-dot"
                        style={{
                          // UPDATED: vw/vh
                          top: `${(m.y / vh) * 100}%`,
                          left: `${(m.x / vw) * 100}%`,
                        }}
                      >
                        <span className="dot-label">{i + 1}</span>
                      </div>
                    ))}
                  </div>

                  <div className="file-info">
                    <div className="file-name">{selectedFile.name}</div>
                    <div className="file-details">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB •{" "}
                      {isImage ? "Image" : "Video"}
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

export default WrongWay;
