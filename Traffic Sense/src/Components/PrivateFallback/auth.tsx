export const setToken = (token:string) => {
  localStorage.setItem('login_auth_token', token);
};

export const getToken = () => {
  return localStorage.getItem('login_auth_token');
};

export const removeToken = () => {
  localStorage.removeItem('login_auth_token');
};
