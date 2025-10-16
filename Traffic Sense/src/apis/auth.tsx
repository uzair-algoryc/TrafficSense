/* eslint-disable @typescript-eslint/no-explicit-any */
//Login API
export const handleLogin = async (email: string, password: string) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BASE_URL}/api/users/login`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      }
    );

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Error logging in:", error);
    return { success: false, message: "Network error" };
  }
};

//Forgot Email API
export const handleForgot = async (email: string) => {
  return fetch(`${import.meta.env.VITE_BASE_URL}/api/accounts/password-reset-request/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email }),
  })
    .then((response) => response.json())
    .then((data) => {
      return data;
    })
    .catch((error) => {
      console.error("Forgot Password Error:", error);
    });
}

//Forgot Password API
export const handleReset = async (token:any, password: string, confirmPassword: string) => {
  return fetch(`${import.meta.env.VITE_BASE_URL}/api/accounts/password-reset-confirm/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ password,confirmPassword, token}),
  })
    .then((response) => response.json())
    .then((data) => {
      return data;
    })
    .catch((error) => {
      console.error("Reset Password Error:", error);
    });
}
