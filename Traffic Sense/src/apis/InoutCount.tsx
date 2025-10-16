export const runInOutCountAnalysis = async (formData: FormData) => {

  const response = await fetch(`${import.meta.env.VITE_BASE_URL}/count_vehicles`, { // Assuming endpoint; adjust as needed
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  return response.json();
};