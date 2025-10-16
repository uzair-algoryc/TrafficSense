/* eslint-disable @typescript-eslint/no-explicit-any */
export const runAINumberPlate = async (formData: FormData, file: File) => {
  try {
    // Decide endpoint based on file type
    const isVideo = file.type.startsWith("video/");
    const endpoint = isVideo ? "/alpr_video" : "/alpr_image";

    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}${endpoint}`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Request failed");
    }

    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error("API Error:", error);
    throw error;
  }
};
