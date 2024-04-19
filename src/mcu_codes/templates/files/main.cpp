/**
  **************************************************************************
  * @file     main.c
  * @brief    main program
  **************************************************************************
  *                       Copyright notice & Disclaimer
  *
  * The software Board Support Package (BSP) that is made available to
  * download from Artery official website is the copyrighted work of Artery.
  * Artery authorizes customers to use, copy, and distribute the BSP
  * software and its related documentation for the purpose of design and
  * development in conjunction with Artery microcontrollers. Use of the
  * software is governed by this copyright notice and the following disclaimer.
  *
  * THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS WITHOUT WARRANTIES,
  * GUARANTEES OR REPRESENTATIONS OF ANY KIND. ARTERY EXPRESSLY DISCLAIMS,
  * TO THE FULLEST EXTENT PERMITTED BY LAW, ALL EXPRESS, IMPLIED OR
  * STATUTORY OR OTHER WARRANTIES, GUARANTEES OR REPRESENTATIONS,
  * INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.
  *
  **************************************************************************
  */

#include "at32f435_437_board.h"
#include "at32f435_437_clock.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "sharedData.h"
#include "SoundRecording.h"

/** @addtogroup AT32F435_periph_template
  * @{
  */

/** @addtogroup 435_LED_toggle LED_toggle
  * @{
  */

//#define SKIP_NEURAL_NETWORK		1


#define DELAY                            100
#define FAST                             1
#define SLOW                             4
#define MS_TICK                          (system_core_clock / 1000U)
uint8_t g_speed = FAST;

void button_exint_init(void);
void button_isr(void);

struct RetResult PredictResult;
uint32_t soundNumber = 0;
volatile uint32_t ticks = 0;


#define SOUND_DETECTION_IDLE   0
#define SOUND_DETECTION_ACTIVE 1
uint8_t soundDetectionState = SOUND_DETECTION_IDLE;

// This is the audio buffers passed to the neural network to analyze
static int16_t   soundBuffer0[SOUND_BUFFER_LENGTH];
static int16_t   soundBuffer1[SOUND_BUFFER_LENGTH];
int16_t * pSoundBuffer[2]; 
uint8_t currActiveBuffer; // help to indicate which soundBuffer is active for neural network, the other will be used to store inputs
uint8_t currStoreBuffer;

extern "C" void SoundPostProcessing(void);
extern "C" void audio_recording_setup(void);
extern "C" void audio_recording_start(void);
extern "C" void audio_recording_stop(void);
extern void soundsetup();
extern int8_t * analysissound(uint32_t sound_number, int8_t* sounds, uint32_t sound_length);
extern __IO bool newBufferReady;
extern __IO bool PingPongFull;
/**
  * @brief  configure button exint
  * @param  none
  * @retval none
  */
void button_exint_init(void)
{
  exint_init_type exint_init_struct;

  crm_periph_clock_enable(CRM_SCFG_PERIPH_CLOCK, TRUE);
  scfg_exint_line_config(SCFG_PORT_SOURCE_GPIOA, SCFG_PINS_SOURCE0);

  exint_default_para_init(&exint_init_struct);
  exint_init_struct.line_enable = TRUE;
  exint_init_struct.line_mode = EXINT_LINE_INTERRUPUT;
  exint_init_struct.line_select = EXINT_LINE_0;
  exint_init_struct.line_polarity = EXINT_TRIGGER_RISING_EDGE;
  exint_init(&exint_init_struct);

  nvic_priority_group_config(NVIC_PRIORITY_GROUP_4);
  nvic_irq_enable(EXINT0_IRQn, 0, 0);
}

/**
  * @brief  button handler function
  * @param  none
  * @retval none
  */
void button_isr(void)
{
  /* delay 5ms */
  delay_ms(5);

  /* clear interrupt pending bit */
  exint_flag_clear(EXINT_LINE_0);

  /* check input pin state */
  if(SET == gpio_input_data_bit_read(USER_BUTTON_PORT, USER_BUTTON_PIN))
  {
    if(g_speed == SLOW)
      g_speed = FAST;
    else
      g_speed = SLOW;
  }
}

/**
  * @brief  exint0 interrupt handler
  * @param  none
  * @retval none
  */
void EXINT0_IRQHandler(void)
{
  button_isr();
}

/**
  * @brief  main function.
  * @param  none
  * @retval none
  */



int main(void)
{
	uint32_t soundLength = SOUND_BUFFER_LENGTH;
	uint32_t i;
	int8_t *returnValue;
	
	currStoreBuffer = 0;
	currActiveBuffer = 1;

	
  system_clock_config();
//	delay_init();
	uart_print_init(115200);
	
	printf("Sound Analysis with TFLite\n");
	

	pSoundBuffer[0] = soundBuffer0;
	pSoundBuffer[1] = soundBuffer1;
	/* config systick clock source */
  systick_clock_source_config(SYSTICK_CLOCK_SOURCE_AHBCLK_NODIV);

  /* config systick reload value and enable interrupt */
  SysTick_Config(MS_TICK);
	
  at32_board_init();
	at32_led_off(LED2);
	at32_led_off(LED3);
	at32_led_off(LED4);
	
//  button_exint_init();

	audio_recording_setup();
	printf("audio recording setup done\n");

#if defined(SKIP_NEURAL_NETWORK)

#else
	soundsetup();
	PredictResult.inputNumber = 0;
	PredictResult.max_idx = 0;
	PredictResult.max_value = -128;
#endif

	printf("start audio detection\n");
	audio_recording_start();
	
  while(1)
  {
/*		
		if (at32_button_press() == USER_BUTTON)
		{
			printf("button pressed\n");
			if (soundDetectionState == SOUND_DETECTION_IDLE)
			{
				printf("start detection\n");
				audio_recording_start();
			}	
			else
			{
				printf("stop detection\n");
				audio_recording_stop();
			}
		}
		else
*/		
		{

			if (newBufferReady)
			{
				ticks = 0;
				SoundPostProcessing();
				printf("[%d] post process audio buffer %d takes ticks %d.\n", soundNumber, currActiveBuffer, ticks);
				ticks = 0;
#if defined(SKIP_NEURAL_NETWORK)		
				delay_ms(500);				
#else
				returnValue = analysissound(soundNumber, (int8_t *)pSoundBuffer[currActiveBuffer], soundLength);
#endif			
				printf("[%d] process audio buffer %d takes ticks %d, dim 0 = %d, dim 1 = %d.\n", soundNumber, currActiveBuffer, ticks, 
					returnValue[0], returnValue[1]);
				if ((returnValue != nullptr) && ((returnValue[0] > returnValue[1]) && (returnValue[0]> 30)))
				{
					at32_led_on(LED2);
				}
				else
				{
					at32_led_off(LED2);
				}
				soundNumber++;
				newBufferReady = false;
			}			
			
		}
		

    
		
    /*
    at32_led_toggle(LED2);
    delay_ms(g_speed * DELAY);
    at32_led_toggle(LED3);
    delay_ms(g_speed * DELAY);
    at32_led_toggle(LED4);
    delay_ms(g_speed * DELAY);
    */
  }
}

/**
  * @}
  */

/**
  * @}
  */
