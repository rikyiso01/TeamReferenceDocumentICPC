FROM alpine

WORKDIR /app
RUN apk add --no-cache pandoc
COPY reference.md ./
CMD ["sh","-c","pandoc reference.md -o /out/reference.html -s --section-divs --katex"]
