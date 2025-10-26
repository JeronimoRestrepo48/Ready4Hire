FROM mcr.microsoft.com/playwright:v1.40.1-focal

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy test files
COPY . .

# Install browsers
RUN npx playwright install

CMD ["npx", "playwright", "test"]

