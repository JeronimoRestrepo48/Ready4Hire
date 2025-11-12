/**
 * Interview Redux Slice
 * Handles interview state and actions
 */

import {createSlice, createAsyncThunk, PayloadAction} from '@reduxjs/toolkit';
import {apiClient} from '../../services/api/ApiClient';
import {
  InterviewState,
  Interview,
  InterviewResponse,
  Question,
  StartInterviewRequest,
  ProcessAnswerRequest,
} from '../../types';

const initialState: InterviewState = {
  currentInterview: null,
  interviews: [],
  loading: false,
  error: null,
};

// Start Interview Thunk
export const startInterview = createAsyncThunk(
  'interview/start',
  async (request: StartInterviewRequest, {rejectWithValue}) => {
    try {
      const response = await apiClient.post<InterviewResponse>('/interviews', request);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  }
);

// Process Answer Thunk
export const processAnswer = createAsyncThunk(
  'interview/processAnswer',
  async ({interviewId, answer, timeTaken}: ProcessAnswerRequest, {rejectWithValue}) => {
    try {
      const response = await apiClient.post<InterviewResponse>(
        `/interviews/${interviewId}/answers`,
        {answer, timeTaken}
      );
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  }
);

// End Interview Thunk
export const endInterview = createAsyncThunk(
  'interview/end',
  async (interviewId: string, {rejectWithValue}) => {
    try {
      const response = await apiClient.post(`/interviews/${interviewId}/end`);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message);
    }
  }
);

// Slice
const interviewSlice = createSlice({
  name: 'interview',
  initialState,
  reducers: {
    setCurrentInterview: (state, action: PayloadAction<Interview | null>) => {
      state.currentInterview = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Start Interview
    builder
      .addCase(startInterview.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(startInterview.fulfilled, (state, action) => {
        state.loading = false;
        const res = action.payload as any;
        state.currentInterview = {
          id: res.interviewId,
          userId: '',
          role: '',
          skillLevel: 'mid',
          interviewType: 'technical',
          currentPhase: res.phase || 'context',
          status: res.status as any,
          questions: res.firstQuestion ? [res.firstQuestion.id] : [],
          answers: [],
          contextAnswers: [],
          currentQuestionIndex: 0,
          createdAt: new Date().toISOString(),
        } as any;
      })
      .addCase(startInterview.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Process Answer
    builder
      .addCase(processAnswer.pending, (state) => {
        state.loading = true;
      })
      .addCase(processAnswer.fulfilled, (state, action) => {
        state.loading = false;
        // Advance progress
        if (state.currentInterview) {
          state.currentInterview.currentQuestionIndex += 1;
          const next = (action.payload as any).nextQuestion as any;
          if (next) {
            state.currentInterview.questions.push(next.id);
          }
        }
      })
      .addCase(processAnswer.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // End Interview
    builder
      .addCase(endInterview.fulfilled, (state) => {
        state.currentInterview = null;
      });
  },
});

export const {setCurrentInterview, clearError} = interviewSlice.actions;
export default interviewSlice.reducer;

