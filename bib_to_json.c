#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "cJSON.h" // Include the cJSON header

// --- Global counters for statistics ---
int total_entries_processed = 0;
int valid_entries_converted = 0;
int disregarded_entries_count = 0;

// --- Helper function to skip whitespace and comments (%) ---
void skip_whitespace_and_comments(FILE *f) {
    int c;
    while ((c = getc(f)) != EOF) {
        if (isspace(c)) {
            continue; // Skip whitespace
        }
        if (c == '%') { // Skip comments until newline
            while ((c = getc(f)) != EOF && c != '\n');
            continue; // Continue skipping whitespace after comment
        }
        ungetc(c, f); // Put the non-whitespace/non-comment character back
        break;
    }
}

// --- Helper function to remove newline characters from a string ---
// Returns a new string with newlines replaced by spaces. Caller must free.
char* remove_newlines(const char *input) {
    if (!input) return NULL;
    size_t len = strlen(input);
    char *output = (char*)malloc(len + 1);
    if (!output) {
        perror("Failed to allocate memory for removing newlines");
        return NULL;
    }
    size_t j = 0;
    for (size_t i = 0; i < len; ++i) {
        if (input[i] == '\n' || input[i] == '\r') {
            if (j > 0 && output[j-1] != ' ') { // Avoid multiple spaces
                 output[j++] = ' ';
            } else if (j == 0) { // Skip leading newlines
                continue;
            }
        } else {
            output[j++] = input[i];
        }
    }
    output[j] = '\0';
    return output;
}


// --- Helper function to read until a specific character (or whitespace for key/type) ---
// Reads alphanumeric + some symbols for key/type, or anything within delimiters for value
// This version is slightly improved but still simplified for BibTeX values.
char* read_until_delimiter(FILE *f, const char delimiter_chars, int is_value) {
    int c;
    size_t buffer_size = 128; // Start with a slightly larger buffer
    char *buffer = (char*)malloc(buffer_size);
    if (!buffer) {
        perror("Failed to allocate buffer");
        return NULL;
    }
    size_t len = 0;
    int inside_quotes = 0;
    int brace_level = 0;
    int escaped = 0;

    // If reading a value, read the opening delimiter
    if (is_value) {
         c = getc(f); // Should be '{' or '"'
         if (c == EOF) { free(buffer); return NULL; } // Unexpected EOF
         if (c == '"') inside_quotes = 1;
         else if (c == '{') brace_level = 1;
         else { // Unexpected character, put it back and fail
             ungetc(c, f);
             free(buffer);
             return NULL; // Expected { or " for value
         }
         // Don't add the opening delimiter to the buffer
    }


    while ((c = getc(f)) != EOF) {
        // Handle escaped characters *within* the value
        if (escaped) {
            escaped = 0;
            // Store escaped character (BibTeX escaping is complex, this is basic)
             if (len + 1 >= buffer_size) {
                 buffer_size *= 2;
                 char *new_buffer = (char*)realloc(buffer, buffer_size);
                 if (!new_buffer) { perror("Realloc failed"); free(buffer); return NULL; }
                 buffer = new_buffer;
             }
             buffer[len++] = c;
             continue;
        }

        if (c == '\\') {
            escaped = 1;
            continue;
        }

        if (is_value) {
            if (inside_quotes) {
                if (c == '"') { // End of value (unless escaped)
                     if (!escaped) {
                         if (len + 1 >= buffer_size) {
                             buffer_size *= 2;
                             char *new_buffer = (char*)realloc(buffer, buffer_size);
                             if (!new_buffer) { perror("Realloc failed"); free(buffer); return NULL; }
                             buffer = new_buffer;
                         }
                         buffer[len] = '\0';
                         return buffer;
                     }
                }
            } else if (brace_level > 0) {
                if (c == '{') brace_level++;
                if (c == '}') {
                    brace_level--;
                    if (brace_level == 0) { // End of value
                         if (len + 1 >= buffer_size) {
                             buffer_size *= 2;
                             char *new_buffer = (char*)realloc(buffer, buffer_size);
                             if (!new_buffer) { perror("Realloc failed"); free(buffer); return NULL; }
                             buffer = new_buffer;
                         }
                         buffer[len] = '\0';
                         return buffer;
                    }
                }
            } else {
                 // Should not happen if is_value is true and we read '{' or '"' initially
                 ungetc(c, f); // Put it back
                 break; // Error or end condition not handled
            }
        } else { // Reading key or type
             if (c == delimiter_chars || isspace(c)) { // Use delimiter_chars for keys/types
                 ungetc(c, f); // Put the delimiter/space back
                 break; // End of key/type
             }
             // Simple check for characters allowed in keys/types (highly simplified)
             if (!isalnum(c) && strchr("_-:.+", c) == NULL) {
                 // fprintf(stderr, "Warning: Unexpected character '%c' (%d) while reading key/type. May cause issues.\n", c, c);
             }
        }

        // Add character to buffer
        if (len + 1 >= buffer_size) {
            buffer_size *= 2;
            char *new_buffer = (char*)realloc(buffer, buffer_size);
            if (!new_buffer) { perror("Realloc failed"); free(buffer); return NULL; }
            buffer = new_buffer;
        }
        buffer[len++] = c;
    }

    // If loop finished without finding end delimiter (shouldn't happen in a valid entry)
    if (is_value && (inside_quotes || brace_level > 0)) {
        fprintf(stderr, "Error: Unexpected EOF or parsing issue while reading value. Buffer content: '%.*s'\n", (int)len, buffer);
        free(buffer);
        return NULL; // Did not find closing quote or brace
    }

    buffer[len] = '\0';
    return buffer;
}


// --- Function to parse a single BibTeX entry from file ---
// Returns a cJSON object for the entry, or NULL on EOF, or cJSON_CreateNull() on parsing error for an entry.
cJSON* parse_bib_entry(FILE *f, char *entry_type_out, size_t type_buffer_size) {
    skip_whitespace_and_comments(f);

    // Check for the start of an entry
    int c = getc(f);
    if (c == EOF) return NULL; // End of file
    if (c != '@') {
        fprintf(stderr, "Warning: Expected '@', found '%c' (%d). Attempting to resync.\n", c, c);
        // Try to resync by finding the next '@' or EOF
        while ((c = getc(f)) != EOF && c != '@');
        if (c == '@') ungetc(c, f); // Found the next entry start
        disregarded_entries_count++; // Count this as a disregarded entry
        return cJSON_CreateNull(); // Indicate a skipped invalid entry
    }

    // Read entry type
    skip_whitespace_and_comments(f);
    char *entry_type_raw = read_until_delimiter(f, '{', 0); // Read until '{'
    if (!entry_type_raw) {
         fprintf(stderr, "Error: Failed to read entry type after '@'. Attempting to resync.\n");
         // Try to find the end of the current block based on brace balance (simplified)
         int level = 0;
         while((c = getc(f)) != EOF) {
             if (c == '{') level++;
             if (c == '}') {
                 if (level == 0) break; // Found closing brace for the entry
                 level--;
             }
         }
         disregarded_entries_count++; // Count this as a disregarded entry
         return cJSON_CreateNull(); // Indicate a skipped invalid entry
    }
    strncpy(entry_type_out, entry_type_raw, type_buffer_size - 1);
    entry_type_out[type_buffer_size - 1] = '\0';
    // Convert entry type to lowercase for consistency in JSON
    for(char *p = entry_type_out; *p; ++p) *p = tolower(*p);
    free(entry_type_raw);
    entry_type_raw = NULL; // Prevent use after free

    // Expect opening brace '{'
    skip_whitespace_and_comments(f);
    c = getc(f);
    if (c != '{') {
        fprintf(stderr, "Error: Expected '{' after entry type '%s', found '%c' (%d). Attempting to resync.\n", entry_type_out, c, c);
         // Try to find the end of the entry based on brace balance (simplified)
         int level = 0;
         while((c = getc(f)) != EOF) {
             if (c == '{') level++;
             if (c == '}') {
                 if (level == 0) break; // Found closing brace for the entry
                 level--;
             }
         }
        disregarded_entries_count++; // Count this as a disregarded entry
        return cJSON_CreateNull(); // Indicate a skipped invalid entry
    }

    // Read entry key
    skip_whitespace_and_comments(f);
    char *entry_key = read_until_delimiter(f, ',', 0); // Read until ','
     if (!entry_key) {
        fprintf(stderr, "Error: Failed to read entry key for entry type '%s'. Attempting to resync.\n", entry_type_out);
        // Try to find the end of the entry based on brace balance (simplified)
         int level = 0;
         while((c = getc(f)) != EOF) {
             if (c == '{') level++;
             if (c == '}') {
                 if (level == 0) break; // Found closing brace for the entry
                 level--;
             }
         }
        disregarded_entries_count++; // Count this as a disregarded entry
        return cJSON_CreateNull(); // Indicate a skipped invalid entry
     }

    cJSON *entry_json = cJSON_CreateObject();
    if (!entry_json) {
        perror("Failed to create JSON object");
        free(entry_key);
        // No resync needed, this is a memory allocation failure
        return NULL; // Indicate a critical error
    }

    // Add entry type and key to JSON
    cJSON_AddStringToObject(entry_json, "ENTRYTYPE", entry_type_out);
    cJSON_AddStringToObject(entry_json, "ID", entry_key);
    free(entry_key);
    entry_key = NULL; // Prevent use after free

    // Read fields
    while (1) {
        skip_whitespace_and_comments(f);
        c = getc(f);
        if (c == EOF) {
            fprintf(stderr, "Error: Unexpected EOF inside entry '%s'.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
            cJSON_Delete(entry_json);
            disregarded_entries_count++; // Count as disregarded
            return cJSON_CreateNull(); // Indicate a disregarded entry due to EOF
        }
        if (c == '}') { // End of entry
            break;
        }
        if (c == ',') { // Separator, continue to next field
            skip_whitespace_and_comments(f);
            c = getc(f);
             if (c == EOF) {
                fprintf(stderr, "Error: Unexpected EOF after comma inside entry '%s'.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
                cJSON_Delete(entry_json);
                disregarded_entries_count++; // Count as disregarded
                return cJSON_CreateNull(); // Indicate a disregarded entry due to EOF
            }
            if (c == '}') { // Trailing comma before closing brace
                 break;
            }
            ungetc(c, f); // Put back the start of the next field
            continue;
        }
        ungetc(c, f); // Put back the start of the field name

        // Read field name
        char *field_name = read_until_delimiter(f, '=', 0); // Read until '='
         if (!field_name) {
            fprintf(stderr, "Warning: Failed to read field name for entry '%s'. Skipping problematic part.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
             // Attempt to skip until the next comma or closing brace
             int level = 0; // Assuming we are at field level
             while((c = getc(f)) != EOF) {
                 if (c == '{') level++;
                 if (c == '}') {
                     if (level == 0) break; // Found closing brace for the entry
                     level--;
                 }
                 if (c == ',' && level == 0) break; // Found field separator
             }
             if (c == EOF) {
                 fprintf(stderr, "Error: Unexpected EOF while skipping problematic field in entry '%s'.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
                 cJSON_Delete(entry_json);
                 disregarded_entries_count++; // Count as disregarded
                 return cJSON_CreateNull(); // Indicate a disregarded entry
             }
             if (c == '}') ungetc(c, f); // Put back the closing brace if found
             continue; // Continue parsing the rest of the entry
         }
        skip_whitespace_and_comments(f);

        // Expect '='
        c = getc(f);
        if (c != '=') {
            fprintf(stderr, "Warning: Expected '=' after field name '%s' in entry '%s', found '%c' (%d). Skipping problematic part.\n", field_name, cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID", c, c);
            free(field_name);
             // Attempt to skip until the next comma or closing brace
             int level = 0; // Assuming we are at field level
             while((c = getc(f)) != EOF) {
                 if (c == '{') level++;
                 if (c == '}') {
                     if (level == 0) break; // Found closing brace for the entry
                     level--;
                 }
                 if (c == ',' && level == 0) break; // Found field separator
             }
              if (c == EOF) {
                 fprintf(stderr, "Error: Unexpected EOF while skipping problematic field in entry '%s'.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
                 cJSON_Delete(entry_json);
                 disregarded_entries_count++; // Count as disregarded
                 return cJSON_CreateNull(); // Indicate a disregarded entry
             }
             if (c == '}') ungetc(c, f); // Put back the closing brace if found
            continue; // Continue parsing the rest of the entry
        }
        skip_whitespace_and_comments(f);

        // Read field value
        char *field_value = read_until_delimiter(f, '\0', 1); // Read value inside {} or ""
         if (!field_value) {
            fprintf(stderr, "Warning: Failed to read value for field '%s' in entry '%s'. Skipping problematic part.\n", field_name, cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
            free(field_name);
             // Attempt to skip until the next comma or closing brace
             int level = 0; // Assuming we were trying to read a value
             while((c = getc(f)) != EOF) {
                 if (c == '{') level++;
                 if (c == '}') {
                     if (level == 0) break; // Found closing brace for the entry
                     level--;
                 }
                 if (c == ',' && level == 0) break; // Found field separator
             }
              if (c == EOF) {
                 fprintf(stderr, "Error: Unexpected EOF while skipping problematic field in entry '%s'.\n", cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
                 cJSON_Delete(entry_json);
                 disregarded_entries_count++; // Count as disregarded
                 return cJSON_CreateNull(); // Indicate a disregarded entry
             }
             if (c == '}') ungetc(c, f); // Put back the closing brace if found
            continue; // Continue parsing the rest of the entry
         }

        // Remove newlines from the field value
        char *cleaned_value = remove_newlines(field_value);
        free(field_value); // Free the original value
        field_value = cleaned_value; // Use the cleaned value

        // Add field to JSON object
        if (field_value) { // Check if cleaning was successful
             cJSON_AddStringToObject(entry_json, field_name, field_value);
        } else {
             fprintf(stderr, "Warning: Failed to clean value for field '%s' in entry '%s'. Skipping field.\n", field_name, cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
        }


        // Clean up
        free(field_name);
        free(field_value); // Free the cleaned value
        field_name = NULL;
        field_value = NULL; // Prevent use after free
    }

    // If we reached here, the entry was parsed successfully (even if some fields were skipped)
    valid_entries_converted++;
    return entry_json; // Return the parsed entry as a cJSON object
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_bib_file>\n", argv[0]);
        return 1;
    }
    const char *input_filename = argv[1];

    // Use a fixed output filename
    const char *output_filename = "data/corpus.json";

    FILE *bib_file;
    FILE *json_file;

    char entry_type[64]; // Buffer to store entry type

    // Data structures for statistics
    cJSON *year_counts = cJSON_CreateObject(); // JSON object to store year counts
    cJSON *all_key_counts = cJSON_CreateObject(); // For general key statistics

    if (!year_counts || !all_key_counts) {
        perror("Failed to create JSON objects for statistics");
        if (year_counts) cJSON_Delete(year_counts);
        if (all_key_counts) cJSON_Delete(all_key_counts);
        return 1;
    }


    // Open input BibTeX file
    bib_file = fopen(input_filename, "r");
    if (bib_file == NULL) {
        perror("Error opening input BibTeX file");
        cJSON_Delete(year_counts);
        cJSON_Delete(all_key_counts);
        return 1;
    }

    // Open output JSON file for converted entries
    json_file = fopen(output_filename, "w");
    if (json_file == NULL) {
        perror("Error opening output JSON file");
        fclose(bib_file);
        cJSON_Delete(year_counts);
        cJSON_Delete(all_key_counts);
        return 1;
    }

    cJSON *json_root = cJSON_CreateArray(); // Create a JSON array to hold entries
    if (!json_root) {
        perror("Failed to create JSON root array");
        fclose(bib_file);
        fclose(json_file);
        cJSON_Delete(year_counts);
        cJSON_Delete(all_key_counts);
        return 1;
    }

    printf("Starting conversion from %s to %s...\n", input_filename, output_filename);

    // Parse entries one by one
    cJSON *entry_json;
    while ((entry_json = parse_bib_entry(bib_file, entry_type, sizeof(entry_type))) != NULL) {
        total_entries_processed++;
        if (!cJSON_IsNull(entry_json)) {
             cJSON_AddItemToArray(json_root, entry_json);
             // valid_entries_converted is incremented inside parse_bib_entry

             // --- Collect Statistics for Yearly Paper Counts CSV ---
             cJSON *year_item = cJSON_GetObjectItemCaseSensitive(entry_json, "year");
             if (year_item && cJSON_IsString(year_item)) {
                 const char *year_str = year_item->valuestring;
                 // Check if year is a valid number (basic check)
                 int is_numeric_year = 1;
                 for (const char *p = year_str; *p; ++p) {
                     if (!isdigit(*p)) {
                         is_numeric_year = 0;
                         break;
                     }
                 }
                 if (is_numeric_year && strlen(year_str) > 0) {
                     cJSON *count_item = cJSON_GetObjectItemCaseSensitive(year_counts, year_str);
                     if (count_item) {
                         // Year already exists, increment count
                         cJSON_SetNumberValue(count_item, count_item->valuedouble + 1);
                     } else {
                         // First entry for this year, add with count 1
                         cJSON_AddNumberToObject(year_counts, year_str, 1);
                     }
                 } else {
                      fprintf(stderr, "Warning: Invalid year format '%s' in entry '%s'. Skipping year count for this entry.\n", year_str, cJSON_GetObjectItemCaseSensitive(entry_json, "ID") ? cJSON_GetObjectItemCaseSensitive(entry_json, "ID")->valuestring : "Unknown ID");
                 }
             }

             // --- Collect General Key Statistics ---
             cJSON *current_field = entry_json->child;
             while (current_field) {
                 // Don't count internally used keys like ENTRYTYPE or ID for these general stats
                 if (strcmp(current_field->string, "ENTRYTYPE") != 0 && strcmp(current_field->string, "ID") != 0) {
                     cJSON *key_count_item = cJSON_GetObjectItemCaseSensitive(all_key_counts, current_field->string);
                     if (key_count_item) {
                         cJSON_SetNumberValue(key_count_item, key_count_item->valuedouble + 1);
                     } else {
                         cJSON_AddNumberToObject(all_key_counts, current_field->string, 1);
                     }
                 }
                 current_field = current_field->next;
             }

        } else {
             cJSON_Delete(entry_json); // Delete the null object indicating a disregarded entry
             // disregarded_entries_count is incremented inside parse_bib_entry
        }

        if (total_entries_processed % 1000 == 0) {
            printf("Processed %d entries...\n", total_entries_processed);
        }
    }

    printf("\nConversion statistics:\n");
    printf("Total entries processed: %d\n", total_entries_processed);
    printf("Valid entries converted: %d\n", valid_entries_converted);
    printf("Entries disregarded (parsing errors): %d\n", disregarded_entries_count);

    // --- Print General Key Statistics ---
    printf("\nField occurrence percentages (for valid entries):\n");
    cJSON *current_key_stat = all_key_counts->child;
    while (current_key_stat) {
        double percent = 0.0;
        if (valid_entries_converted > 0) {
            percent = (current_key_stat->valuedouble / valid_entries_converted) * 100.0;
        }
        printf("  %s: %.0f (%.2f%%)\n", current_key_stat->string, current_key_stat->valuedouble, percent);
        current_key_stat = current_key_stat->next;
    }
    printf("\n");

    // Write the main JSON array to the output file
    char *json_string = cJSON_Print(json_root); // Use cJSON_Print for formatted output
    // char *json_string = cJSON_PrintUnformatted(json_root); // Use this for smaller output size
    if (json_string == NULL) {
        perror("Failed to print main JSON");
        // Clean up will happen below
    } else {
        fprintf(json_file, "%s\n", json_string);
        free(json_string);
    }

    // Clean up cJSON objects
    cJSON_Delete(json_root);
    cJSON_Delete(year_counts);
    cJSON_Delete(all_key_counts);

    // Close files
    fclose(bib_file);
    fclose(json_file);

    printf("Conversion and statistics generation complete.\n");

    return 0;
}
