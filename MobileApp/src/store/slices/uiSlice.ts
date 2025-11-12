/**
 * UI Redux Slice
 */

import {createSlice, PayloadAction} from '@reduxjs/toolkit';
import {UIState} from '../../types';

const initialState: UIState = {
  theme: 'light',
  language: 'es',
  notifications: true,
  offlineMode: false,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
    },
    setLanguage: (state, action: PayloadAction<'es' | 'en' | 'pt' | 'fr'>) => {
      state.language = action.payload;
    },
    setNotifications: (state, action: PayloadAction<boolean>) => {
      state.notifications = action.payload;
    },
    setOfflineMode: (state, action: PayloadAction<boolean>) => {
      state.offlineMode = action.payload;
    },
  },
});

export const {setTheme, setLanguage, setNotifications, setOfflineMode} = uiSlice.actions;
export default uiSlice.reducer;

