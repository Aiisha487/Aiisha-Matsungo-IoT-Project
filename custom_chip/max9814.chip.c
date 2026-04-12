#include "wokwi-api.h"  // 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// DATASHEET CONSTANTS
// ============================================================================
#define SOUND_MIN 0.0f
#define SOUND_MAX 100.0f
#define SAMPLE_INTERVAL_MS 1000

// ============================================================================
// CHIP STATE
// ============================================================================
typedef struct {
  pin_t out_pin;
  pin_t gain_pin;
  pin_t ar_pin;
  timer_t sample_timer;
  uint32_t simulated_seconds;
} chip_state_t;

// ============================================================================
// MATHEMATICAL MODEL FOR SOUND LEVEL
// ============================================================================
float generate_sound_level(int hour) {
  float base;
  if (hour < 6) base = 5.0f;
  else if (hour < 9) base = 30.0f;
  else if (hour < 17) base = 10.0f;
  else if (hour < 21) base = 25.0f;
  else base = 8.0f;

  float ambient = 3.0f * sinf((hour / 24.0f) * 2.0f * M_PI);
  float spike = ((rand() % 100) < 10) ? (float)(rand() % 50 + 30) : 0.0f;
  float noise = ((rand() % 100) - 50) / 25.0f;

  float level = base + ambient + spike + noise;
  if (level < SOUND_MIN) level = SOUND_MIN;
  if (level > SOUND_MAX) level = SOUND_MAX;

  return level;
}

// ============================================================================
// TIMER CALLBACK
// ============================================================================
void on_timer(void *user_data) {
  chip_state_t *state = (chip_state_t *)user_data;
  state->simulated_seconds += 60;
  int hour = (state->simulated_seconds / 3600) % 24;

  float level = generate_sound_level(hour);
  float voltage_normalized = level / 100.0f;
  
  // Update physical pin
  pin_dac_write(state->out_pin, voltage_normalized);

  printf("MAX9814: %02d:00 | Level=%.1f%% | V=%.2fV\n", hour, level, voltage_normalized * 3.3f);
}

// ============================================================================
// CHIP INITIALIZATION
// ============================================================================
void chip_init() {
  chip_state_t *state = malloc(sizeof(chip_state_t));
  if (!state) return; // Safety check

  // Initialize pins
  state->out_pin  = pin_init("SIG",  ANALOG);   
  state->gain_pin = pin_init("GAIN", INPUT);    
  state->ar_pin   = pin_init("AR",   INPUT);    

  srand(42);
  state->simulated_seconds = 0;

  const timer_config_t timer_config = {
    .callback  = on_timer,
    .user_data = state,
  };
  state->sample_timer = timer_init(&timer_config);
  timer_start(state->sample_timer, SAMPLE_INTERVAL_MS * 1000, true);

  printf("MAX9814 Custom Chip initialized on SIG pin\n");
  pin_dac_write(state->out_pin, 0.05f); 
}
