/* eslint-disable @typescript-eslint/no-explicit-any */
export const runAIWrongway = async (formData: FormData) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}/wrong_way_detection`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      // Extract error message from server response
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || "Request failed");
    }

    // ✅ Important: parse JSON response correctly
    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error("API Error:", error);
    throw error;
  }
};
