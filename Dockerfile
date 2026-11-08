FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=3000 DATA_DIR=/data
COPY package.json server.js shared-bank.js world.js ./
COPY public ./public
# Make the app readable by the unprivileged runtime user regardless of the host checkout's umask.
RUN chmod -R a+rX /app
USER node
EXPOSE 3000
HEALTHCHECK --interval=60s --timeout=5s --retries=3 CMD wget -q --spider http://127.0.0.1:3000/healthz || exit 1
CMD ["node", "server.js"]
