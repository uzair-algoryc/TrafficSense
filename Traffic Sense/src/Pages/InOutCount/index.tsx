import React, { useRef, useState, useEffect, type DragEvent } from "react";
import "./style.css";
import { MdOutlineFileUpload } from "react-icons/md";
import { LiaFileVideo } from "react-icons/lia";
import { RxCross2 } from "react-icons/rx";
import { LuSparkles } from "react-icons/lu";
import { BiCheckCircle } from "react-icons/bi";
import { runInOutCountAnalysis } from "../../apis/InoutCount";
import { useToast } from "@chakra-ui/react";

const InOutCount: React.FC = () => {
  const uploadRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null); // NEW: For image dimensions/clicks
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisData, setAnalysisData] = useState<{
    in_count: number;
    out_count: number;
    output_video: string;
  } | null>(null);
  const [markers, setMarkers] = useState<{ x: number; y: number }[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  // NEW: Track dimensions for video/image
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
    const img = imgRef.current;
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
    setAnalysisData(null);
    setVideoDimensions({ width: 0, height: 0 }); // NEW: Reset
    setImgDimensions({ width: 0, height: 0 }); // NEW: Reset
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setIsLoading(false);
    setShowResults(false);
    setAnalysisData(null);
    setMarkers([]);
    setVideoDimensions({ width: 0, height: 0 }); // NEW: Reset
    setImgDimensions({ width: 0, height: 0 }); // NEW: Reset
  };

  const handleRunModel = async () => {
    if (markers.length !== 2) {
      toast({
        title: "Please mark exactly 2 points to define the area (left, right).",
        status: "warning",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      return;
    }

    setIsLoading(true);

    try {
      // ✅ Prepare coordinates
      const coordinatesString = markers
        .map((m) => `${Math.round(m.x)},${Math.round(m.y)}`)
        .join(",");

      // ✅ Create FormData to send file and data
      const formData = new FormData();
      formData.append("file", selectedFile!);
      formData.append("coordinates", coordinatesString);


      const apiPromise = runInOutCountAnalysis(formData);
      const data = await Promise.race([apiPromise]);

      toast({
        title: "Analysis completed successfully!",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top",
      });

      setAnalysisData({
        in_count: data.in_count,
        out_count: data.out_count,
        output_video: data.output_video,
      });
      setShowResults(true);
    } catch (err) {
      console.error("API call failed:", err);
      toast({
        title:
          err instanceof Error ? err.message : "An unexpected error occurred.",
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReRun = () => {
    setShowResults(false);
    setAnalysisData(null);
  };

  const handleNewFile = () => {
    handleRemoveFile();
  };

  const handleMediaClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const isImage = selectedFile?.type.startsWith("image/") || false;
    const mediaElement = isImage
      ? (e.currentTarget.querySelector("img") as HTMLImageElement)
      : (e.currentTarget.querySelector("video") as HTMLVideoElement);

    if (!mediaElement) return;
    if (isImage && !imgRef.current) return; // Guard
    if (!isImage && !videoRef.current) return; // Guard

    const rect = mediaElement.getBoundingClientRect();
    const dimensions = isImage ? imgDimensions : videoDimensions;
    if (dimensions.width === 0) return; // Wait for load

    const x = ((e.clientX - rect.left) / rect.width) * dimensions.width;
    const y = ((e.clientY - rect.top) / rect.height) * dimensions.height;

    if (markers.length < 2) {
      setMarkers((prev) => [...prev, { x, y }]);
    }
  };

  // NEW: Get current dimensions based on type
  const getDimensions = () => {
    const isImage = selectedFile?.type.startsWith("image/") || false;
    return isImage ? imgDimensions : videoDimensions;
  };

  const vw = getDimensions().width || 1;
  const vh = getDimensions().height || 1;

  return (
    <div className="speed-detection-container">
      <div className="page-title">In/Out Count</div>
      <div className="page-subtitle">
        Track and count people or vehicles entering and exiting designated areas.
        Perfect for monitoring foot traffic, occupancy levels, and flow
        analysis.
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
                  <BiCheckCircle className="banner-icon" />
                  <div>
                    <div className="banner-title">In/Out Count Complete</div>
                    <div className="banner-subtitle">
                      Analysis completed successfully using AI model
                    </div>
                  </div>
                </div>

                <div className="results-grid">
                  {/* Left side → Processed output video */}
                  <div className="annotated-output">
                    <div className="section-title">Processed Output</div>
                    <div className="video-wrapper">
                      <video
                        src={`${import.meta.env.VITE_BASE_URL}${analysisData?.output_video || ""}`}
                        controls
                        className="preview-video"
                      />
                    </div>
                  </div>

                  {/* Right side → Summary Metrics */}
                  <div className="summary-metrics">
                    <div className="section-title">Summary Metrics</div>
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <div className="metric-label">In Count</div>
                        <div className="metric-value">
                          {analysisData?.in_count ?? 0}
                        </div>
                      </div>
                      <div className="metric-card">
                        <div className="metric-label">Out Count</div>
                        <div className="metric-value">
                          {analysisData?.out_count ?? 0}
                        </div>
                      </div>
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
                  <div className="media-container" onClick={handleMediaClick}>
                    {selectedFile.type.startsWith("image/") ? (
                      <img ref={imgRef} src={previewURL!} alt="Preview" /> // UPDATED: Added ref
                    ) : (
                      <video ref={videoRef} src={previewURL!} controls />
                    )}

                    {/* SVG lines between markers */}
                    {vw > 0 &&
                      markers.length === 2 && ( // UPDATED: Use vw >0 guard
                        <svg className="lines-overlay">
                          <line
                            x1={`${(markers[0].x / vw) * 100}%`} // UPDATED: Use vw/vh
                            y1={`${(markers[0].y / vh) * 100}%`}
                            x2={`${(markers[1].x / vw) * 100}%`}
                            y2={`${(markers[1].y / vh) * 100}%`}
                            stroke="#3b82f6"
                            strokeWidth="2"
                          />
                        </svg>
                      )}

                    {/* Dots with labels */}
                    {markers.map((m, i) => (
                      <div
                        key={`dot-${i}`}
                        className="marker-dot"
                        style={{
                          // UPDATED: Use vw/vh
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

export default InOutCount;
