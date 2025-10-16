import { createContext } from "react";

export const TitleContextType = createContext<{
    title: string;
    setTitle: (title: string) => void;
}>