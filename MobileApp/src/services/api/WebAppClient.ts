/**
 * WebApp API Client (Blazor/.NET) for Auth endpoints
 */

import axios, {AxiosInstance, AxiosRequestConfig} from 'axios';

const WEBAPP_BASE_URL = process.env.WEBAPP_BASE_URL || 'http://localhost:5214';
const TIMEOUT = 20000;

class WebAppClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${WEBAPP_BASE_URL}/api`,
      timeout: TIMEOUT,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const res = await this.client.post<T>(url, data, config);
    return res.data;
  }
}

export const webAppClient = new WebAppClient();
