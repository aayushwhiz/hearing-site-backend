use futures::future::join_all;
use reqwest::{multipart, Client};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use tokio::fs::File;
use tokio::task;
use tokio_util::io::ReaderStream;
pub async fn split_audio_by_size_and_transcribe(
    input_path: &str,
    max_segment_size: usize,
    openai_api_key: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let base_filename = PathBuf::from(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("Failed to extract file stem")?
        .to_string();

    let split_dir = PathBuf::from("split_audio");
    if !split_dir.exists() {
        std::fs::create_dir_all(&split_dir)?;
    }

    let output_extension = "mp3";
    let segment_duration_secs = max_segment_size / (128000 / 8); // Assuming 128 kbps bitrate
    let total_duration = get_audio_duration(input_path)?; // Assuming you have a function to get the total duration

    let client = Arc::new(Client::new());
    let transcriptions: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(vec![
        None;
        total_segments(
            total_duration,
            segment_duration_secs
        )
    ]));

    let mut all_tasks = vec![]; // To store all tasks (splitting + transcription)

    // Split the audio file into segments and send each segment for transcription concurrently
    for i in 0.. {
        let start_time = i * segment_duration_secs;
        if start_time >= total_duration {
            break;
        }

        let segment_filename = format!("{}_part{}.{}", base_filename, i + 1, output_extension);
        let output_path = split_dir.join(&segment_filename);

        let client_clone = Arc::clone(&client);
        let openai_api_key_clone = openai_api_key.to_string();
        let transcriptions_clone = Arc::clone(&transcriptions);

        let input_path = input_path.to_string();

        // Spawn a task that splits the audio and immediately sends the segment for transcription
        let task = task::spawn(async move {
            // Step 1: Split the audio segment asynchronously
            split_audio_segment(&input_path, start_time, segment_duration_secs, &output_path)?;

            // Step 2: Immediately after splitting, send the segment for transcription
            println!(
                "Sending transcription request for file: {}",
                output_path.display()
            );

            match transcribe_audio_segment(&client_clone, &openai_api_key_clone, &output_path).await
            {
                Ok(transcription) => {
                    let mut transcriptions_lock = transcriptions_clone.lock().unwrap();
                    transcriptions_lock[i] = Some(transcription);
                    println!("Received transcription for file: {}", output_path.display());
                }
                Err(e) => {
                    eprintln!("Error transcribing file {}: {:?}", output_path.display(), e);
                }
            }

            Ok(()) as Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
        });

        // Collect the task handles
        all_tasks.push(task);
    }

    // Wait for all tasks (splitting + transcribing) to complete
    join_all(all_tasks).await;

    // Lock the transcriptions and clone the data safely
    let transcriptions_lock = transcriptions.lock().unwrap();
    let final_transcriptions: Vec<String> = transcriptions_lock
        .clone()
        .into_iter()
        .filter_map(|t| t)
        .collect();

    Ok(final_transcriptions)
}

// Split the audio file into segments
fn split_audio_segment(
    input_path: &str,
    start_time: usize,
    duration_secs: usize,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let status = Command::new("ffmpeg")
        .arg("-i")
        .arg(input_path)
        .arg("-ss")
        .arg(format!("{}", start_time))
        .arg("-t")
        .arg(format!("{}", duration_secs))
        .arg(output_path.to_str().ok_or("Invalid output path")?)
        .stdout(std::process::Stdio::null()) // Suppress stdout
        .stderr(std::process::Stdio::null()) // Suppress stderr
        .status()?;

    if !status.success() {
        return Err("ffmpeg command failed".into());
    }

    Ok(())
}

// Function to get the total duration of the audio file using ffmpeg
fn get_audio_duration(input_path: &str) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let output = Command::new("ffmpeg")
        .arg("-i")
        .arg(input_path)
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()?;

    let output_str = String::from_utf8_lossy(&output.stderr);
    let duration_line = output_str
        .lines()
        .find(|line| line.contains("Duration"))
        .ok_or("Duration not found")?;

    let duration_str = duration_line.split_whitespace().nth(1).unwrap_or("");
    let time_parts: Vec<&str> = duration_str.trim_end_matches(',').split(':').collect();

    if time_parts.len() == 3 {
        let hours: f64 = time_parts[0].parse()?;
        let minutes: f64 = time_parts[1].parse()?;
        let seconds: f64 = time_parts[2].parse()?;

        let total_seconds = hours * 3600.0 + minutes * 60.0 + seconds;
        return Ok(total_seconds as usize);
    }

    Err("Duration not found".into())
}

// Helper to calculate total segments based on duration and segment size
fn total_segments(total_duration: usize, segment_duration_secs: usize) -> usize {
    (total_duration + segment_duration_secs - 1) / segment_duration_secs // Rounds up to the nearest segment
}

// Function to send transcription request to OpenAI
async fn transcribe_audio_segment(
    client: &Client,
    api_key: &str,
    segment_path: &PathBuf,
) -> Result<String, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let audio_file = segment_path.to_str().ok_or("Invalid path")?;
    send_transcription_request(client, api_key, audio_file).await
}

async fn send_transcription_request(
    client: &Client,
    api_key: &str,
    audio_file: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let url = "https://api.openai.com/v1/audio/transcriptions";

    // Open the file asynchronously
    let file = File::open(audio_file).await?;

    // Convert the file into a stream
    let file_stream = ReaderStream::new(file);

    // Create a Part from the stream
    let part = multipart::Part::stream(reqwest::Body::wrap_stream(file_stream))
        .file_name(audio_file.to_string())
        .mime_str("audio/mpeg")?;

    // Build the multipart form
    let form = multipart::Form::new()
        .text("model", "whisper-1")
        .part("file", part);

    // Send the request
    let response = client
        .post(url)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .await?;

    if response.status().is_success() {
        let transcription: serde_json::Value = response.json().await?;
        if let Some(transcription_text) = transcription["text"].as_str() {
            return Ok(transcription_text.to_string());
        }
    }

    Err("Failed to get transcription".into())
}
