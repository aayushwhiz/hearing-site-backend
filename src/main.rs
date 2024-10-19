use actix_cors::Cors;
use actix_multipart::Multipart;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use futures_util::stream::StreamExt as _;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

mod audio_processing;

#[derive(Deserialize)]
struct TranscriptionRequest {
    transcription: String, // This will be the UUID filename
}

// Helper function to read transcription content from the file asynchronously
async fn read_transcription_content(uuid_filename: &str) -> Result<String, std::io::Error> {
    let file_path = if uuid_filename.ends_with(".txt") {
        format!("./transcriptions/{}", uuid_filename)
    } else {
        format!("./transcriptions/{}.txt", uuid_filename)
    };

    let mut file = fs::File::open(file_path).await?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).await?;
    Ok(contents)
}

// Helper function to call OpenAI API with the extracted transcription text
async fn call_openai_api(
    transcription_text: String,
    system_message: &str,
) -> Result<String, reqwest::Error> {
    let client = Client::new();
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let request_body = serde_json::json!({
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": transcription_text
            }
        ]
    });

    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&request_body)
        .send()
        .await;

    match response {
        Ok(successful_response) => {
            let json_response = successful_response.json::<serde_json::Value>().await?;
            let result = json_response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("No response")
                .to_string();
            Ok(result)
        }
        Err(e) => Err(e),
    }
}

// Save result to a file using the same UUID name asynchronously
async fn save_to_file(
    directory: &str,
    uuid_filename: &str,
    content: &str,
) -> Result<(), std::io::Error> {
    let file_path = if uuid_filename.ends_with(".txt") {
        format!("{}/{}", directory, uuid_filename)
    } else {
        format!("{}/{}.txt", directory, uuid_filename)
    };

    let path = std::path::Path::new(&file_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let mut file = fs::File::create(file_path).await?;
    file.write_all(content.as_bytes()).await?;
    Ok(())
}

// Endpoint for generating summary from transcription and returning it
#[post("/summarize")]
async fn summarize(transcription: web::Json<TranscriptionRequest>) -> impl Responder {
    let uuid_filename = &transcription.transcription;

    match read_transcription_content(uuid_filename).await {
        Ok(transcription_text) => {
            let system_message = "Summarize the following transcription...";
            match call_openai_api(transcription_text, system_message).await {
                Ok(summary) => {
                    // Save the generated summary to a file
                    if let Err(e) = save_to_file("./summaries", uuid_filename, &summary).await {
                        return HttpResponse::InternalServerError()
                            .json(json!({"error": format!("Error saving summary: {}", e)}));
                    }
                    // Return the summary in the response
                    HttpResponse::Ok().json(json!({
                        "content": summary
                    }))
                }
                Err(_) => HttpResponse::InternalServerError()
                    .json(json!({"error": "Error generating summary"})),
            }
        }
        Err(_) => HttpResponse::InternalServerError()
            .json(json!({"error": "Error reading transcription"})),
    }
}

// Repeat similar changes for key points, action items, and participants

#[post("/key_points")]
async fn key_points(transcription: web::Json<TranscriptionRequest>) -> impl Responder {
    let uuid_filename = &transcription.transcription;

    match read_transcription_content(uuid_filename).await {
        Ok(transcription_text) => {
            let system_message = "Extract key points from the transcription...";
            match call_openai_api(transcription_text, system_message).await {
                Ok(key_points) => {
                    // Save the generated key points to a file
                    if let Err(e) = save_to_file("./key_points", uuid_filename, &key_points).await {
                        return HttpResponse::InternalServerError()
                            .json(json!({"error": format!("Error saving key points: {}", e)}));
                    }
                    // Return the key points in the response
                    HttpResponse::Ok().json(json!({
                        "content": key_points
                    }))
                }
                Err(_) => HttpResponse::InternalServerError()
                    .json(json!({"error": "Error extracting key points"})),
            }
        }
        Err(_) => HttpResponse::InternalServerError()
            .json(json!({"error": "Error reading transcription"})),
    }
}

// Endpoint for extracting action items from transcription
#[post("/action_items")]
async fn action_items(transcription: web::Json<TranscriptionRequest>) -> impl Responder {
    let uuid_filename = &transcription.transcription;

    match read_transcription_content(uuid_filename).await {
        Ok(transcription_text) => {
            let system_message = "Extract action items from the transcription...";
            match call_openai_api(transcription_text, system_message).await {
                Ok(action_items) => {
                    // Save the generated action items to a file
                    if let Err(e) =
                        save_to_file("./action_items", uuid_filename, &action_items).await
                    {
                        return HttpResponse::InternalServerError()
                            .json(json!({"error": format!("Error saving action items: {}", e)}));
                    }
                    // Return the action items in the response
                    HttpResponse::Ok().json(json!({
                        "content": action_items
                    }))
                }
                Err(_) => HttpResponse::InternalServerError()
                    .json(json!({"error": "Error extracting action items"})),
            }
        }
        Err(_) => HttpResponse::InternalServerError()
            .json(json!({"error": "Error reading transcription"})),
    }
}

// Endpoint for extracting participants from transcription
#[post("/participants")]
async fn participants(transcription: web::Json<TranscriptionRequest>) -> impl Responder {
    let uuid_filename = &transcription.transcription;

    match read_transcription_content(uuid_filename).await {
        Ok(transcription_text) => {
            let system_message = "Extract participants and their details from the transcription...";
            match call_openai_api(transcription_text, system_message).await {
                Ok(participants) => {
                    // Save the generated participants to a file
                    if let Err(e) =
                        save_to_file("./participants", uuid_filename, &participants).await
                    {
                        return HttpResponse::InternalServerError()
                            .json(json!({"error": format!("Error saving participants: {}", e)}));
                    }
                    // Return the participants in the response
                    HttpResponse::Ok().json(json!({
                        "content": participants
                    }))
                }
                Err(_) => HttpResponse::InternalServerError()
                    .json(json!({"error": "Error extracting participants"})),
            }
        }
        Err(_) => HttpResponse::InternalServerError()
            .json(json!({"error": "Error reading transcription"})),
    }
}
#[post("/upload")]
async fn upload_audio(mut payload: Multipart) -> impl Responder {
    // Create a unique filename for the uploaded file
    let uuid = Uuid::new_v4();
    let file_path = format!("./uploads/{}.mp3", uuid);

    // Clone file_path for use inside web::block to avoid lifetime issues
    let file_path_clone = file_path.clone();

    // Save the uploaded file
    let mut file = web::block(move || File::create(&file_path_clone))
        .await
        .expect("Failed to create file for saving the uploaded audio")
        .expect("Failed to open the file");

    // Process each field in the multipart payload
    while let Some(item) = payload.next().await {
        let mut field = item.expect("Failed to process multipart field");

        // Process the field stream
        while let Some(chunk) = field.next().await {
            let data = chunk.expect("Failed to read chunk");

            // Write the chunk to the file
            file = web::block(move || {
                file.write_all(&data)?;
                Ok::<_, std::io::Error>(file)
            })
            .await
            .expect("Failed to write chunk to file")
            .expect("File writing failed");
        }
    }

    // Call the transcription process using the UUID filename
    match process_audio_file(file_path.clone()).await {
        Ok(transcription_filename) => {
            // Return a JSON response instead of plain text
            HttpResponse::Ok().json(serde_json::json!({
                "uploaded_file": file_path,
                "transcription_file": transcription_filename
            }))
        }
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({ "error": format!("Error: {}", e) })),
    }
}

// Download a file from the server
#[get("/download/{category}/{file_name}")]
async fn download_file(path: web::Path<(String, String)>) -> impl Responder {
    let (category, file_name) = path.into_inner();
    let file_path = format!("./{}/{}", category, file_name);

    if let Ok(content) = fs::read(&file_path).await {
        HttpResponse::Ok()
            .content_type("text/plain")
            .insert_header((
                "Content-Disposition",
                format!("attachment; filename={}", file_name),
            ))
            .body(content)
    } else {
        HttpResponse::NotFound().body("File not found")
    }
}

#[get("/health")]
async fn health() -> impl Responder {
    println!("Health check requested");
    HttpResponse::Ok().body("Server is running")
}

async fn process_audio_file(
    file_path: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync + 'static>> {
    println!("Starting transcription process for file: {}", file_path);

    // Load environment variables

    // Get the OpenAI API key from the environment
    let openai_api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    // Debug message for starting transcription process
    println!("API key loaded. Starting the transcription process...");

    // Process and transcribe the audio file using the existing logic
    let transcriptions = audio_processing::split_audio_by_size_and_transcribe(
        &file_path,
        1024 * 1024 * 10, // Example max segment size (5MB)
        &openai_api_key,
    )
    .await?;

    // Debug message for checking if transcriptions were received
    println!("Transcriptions received: {:?}", transcriptions);

    // Combine all the transcriptions into a single line (remove all line breaks)
    let transcription_combined = transcriptions.join(" ");
    println!("Combined transcription: {}", transcription_combined);

    // Ensure the directory exists
    if let Err(e) = std::fs::create_dir_all("./transcriptions") {
        println!("Failed to create directory: {:?}", e);
        return Err(Box::new(e));
    }

    // Generate a unique file name
    let transcription_filename = format!("./transcriptions/{}.txt", Uuid::new_v4());

    // Attempt to create the file
    let mut file = match File::create(&transcription_filename) {
        Ok(f) => f,
        Err(e) => {
            println!("Failed to create file: {:?}", e);
            return Err(Box::new(e));
        }
    };

    // Attempt to write the combined transcription to the file
    if let Err(e) = file.write_all(transcription_combined.as_bytes()) {
        println!("Failed to write to file: {:?}", e);
        return Err(Box::new(e));
    }

    // Debug message to confirm the transcription has been saved
    println!(
        "Transcription successfully written to file: {}",
        transcription_filename
    );

    // Return only the file name, not the full path
    let file_name = Path::new(&transcription_filename)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    Ok(file_name)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_string());

    // Ensure the necessary directories exist
    fs::create_dir_all("./uploads").await?;
    fs::create_dir_all("./transcriptions").await?;
    fs::create_dir_all("./summaries").await?;
    fs::create_dir_all("./key_points").await?;
    fs::create_dir_all("./action_items").await?;
    fs::create_dir_all("./participants").await?;

    // Start the Actix Web server
    HttpServer::new(|| {
        App::new()
            .wrap(
                Cors::permissive(), // This will allow all origins, all methods, all headers
            )
            .service(upload_audio)
            .service(download_file)
            .service(health)
            .service(summarize)
            .service(key_points)
            .service(action_items)
            .service(participants)
    })
    .bind(("0.0.0.0", port.parse().unwrap()))?
    .run()
    .await
}
