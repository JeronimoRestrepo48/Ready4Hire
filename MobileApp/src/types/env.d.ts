/**
 * Environment Variables Type Definitions
 */

declare module '@env' {
  export const API_BASE_URL: string;
  export const API_VERSION: string;
  export const WEBAPP_BASE_URL: string;
  export const ENABLE_PUSH_NOTIFICATIONS: string;
}

// Extend NodeJS.ProcessEnv for process.env usage
declare namespace NodeJS {
  interface ProcessEnv {
    API_BASE_URL?: string;
    API_VERSION?: string;
    WEBAPP_BASE_URL?: string;
    ENABLE_PUSH_NOTIFICATIONS?: string;
  }
}

