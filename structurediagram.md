#
# MDPS System Structure Diagram (Full + Decentralized Data Management)
#
# نسخة كاملة من المخطط مع دمج تعديلات استقلالية إدارة البيانات والتحقق داخل كل قسم، وحذف الإدارة المركزية.

```
MDPS/
│
├── 1. Data_Collection_and_Acquisition/
│   ├── 1.1 data_connectivity_feed_integration/
│   │   ├── 1.1.1 __init__.py
│   │   ├── 1.1.2 bid_ask_streamer.py
│   │   ├── 1.1.3 exchange_api_manager.py
│   │   ├── 1.1.4 historical_data_loader.py
│   │   ├── 1.1.5 live_price_feed.py
│   │   ├── 1.1.6 metatrader5_connector.py
│   │   ├── 1.1.7 mt5_connection.py
│   │   ├── 1.1.8 ohlcv_extractor.py
│   │   ├── 1.1.9 order_book_snapshotter.py
│   │   ├── 1.1.10 tick_data_collector.py
│   │   ├── 1.1.11 volatility_index_tracker.py
│   │   ├── 1.1.12 volume_feed_integrator.py  
│   ├── 1.2 data_manager.py          # إدارة بيانات رئيسية للقسم
│   ├── 1.3 data_storage_profiling/
│   │   ├── 1.3.1 data_buffer_fallback_storage.py
│   │   ├── 1.3.2 data_source_profiler.py
│   │   ├── 1.3.3 raw_data_archiver.py
│   ├── 1.4 data_validation_integrity_assurance/
│   │   ├── 1.4.1 data_anomaly_detector.py
│   │   ├── 1.4.2 feed_integrity_logger.py
│   │   ├── 1.4.3 feed_source_tagger.py
│   │   ├── 1.4.4 live_feed_validator.py
│   ├── 1.5 pipeline_orchestration_monitoring/
│   │   ├── 1.5.1 alert_manager.py
│   │   ├── 1.5.2 data_pipeline_scheduler.py
│   │   ├── 1.5.3 pipeline_monitoring_system.py
│   ├── 1.6 pre_cleaning_preparation/
│   │   ├── 1.6.1 data_sanitizer.py
│   ├── 1.7 time_handling_candle_construction/
│   │   ├── 1.7.1 adaptive_sampling_controller.py
│   │   ├── 1.7.2 candle_constructor.py
│   │   ├── 1.7.3 time_drift_monitor.py
│   │   ├── 1.7.4 time_sync_engine.py
│   ├── 1.8 __init__.py
│   ├── 1.9 validation.py            # تحقق بيانات رئيسي للقسم
│   ├── 1.10 api_interface.py        # واجهة برمجية للتكامل مع الأقسام الأخرى
│   ├── 1.11 event_bus.py            # نظام أحداث داخلي لتبادل الإشعارات والبيانات
│   ├── 1.12 integration_protocols/  # بروتوكولات التكامل مع الأنظمة الخارجية
│   │   ├── 1.12.1 rest_api_protocol.py
│   │   ├── 1.12.2 websocket_protocol.py
│   │   ├── 1.12.3 custom_integration_adapter.py
│   ├── 1.13 monitoring.py           # مراقبة أداء القسم وتسجيل الأحداث
│   ├── 1.14 extensibility.md        # توثيق نقاط التوسع والتكامل مع الأقسام الأخرى
│   # تم إضافة ملفات ونقاط تكامل عملية لزيادة قابلية التوسع والتكامل مع باقي النظام
│
├── 2. External Factors Integration/
│   ├── 2.1 BlockchainAndOnChainAnalytics/
│   │   ├── 2.1.1 BitcoinHashrateAnalyzer.py
│   │   ├── 2.1.2 GeopoliticalRiskIndex.py
│   │   ├── 2.1.3 OnChainDataFetcher.py
│   │   ├── 2.1.4 WhaleActivityTracker.py
│   ├── 2.2 MarketMicrostructureAndCorrelations/
│   │   ├── 2.2.1 CorrelatedAssetTracker.py
│   │   ├── 2.2.2 GoogleTrendsAPIIntegration.py
│   │   ├── 2.2.3 MarketDepthAndOrderBookAnalyzer.py
│   ├── 2.3 NewsAndEconomicEvents/
│   │   ├── 2.3.1 EconomicCalendarIntegrator.py
│   │   ├── 2.3.2 EventImpactEstimator.py
│   │   ├── 2.3.3 HighImpactNewsMapper.py
│   │   ├── 2.3.4 MacroEconomicIndicatorFeed.py
│   │   ├── 2.3.5 NewsSentimentAnalyzer.py
│   ├── 2.4 SocialAndCryptoSentiment/
│   │   ├── 2.4.1 FearAndGreedIndexReader.py
│   │   ├── 2.4.2 FundingRateMonitor.py
│   │   ├── 2.4.3 SentimentAggregator.py
│   │   ├── 2.4.4 SocialMediaSentimentTracker.py
│   │   ├── 2.4.5 TwitterCryptoSentimentScraper.py
│   ├── 2.5 TimeWeightedEventImpactModel/
│   │   ├── 2.5.1 EventImpactTimeDecayModel.py
│   │   ├── 2.5.2 ImpactWeightCalculator.py
│   ├── 2.6 data_manager.py          # إدارة بيانات محلية للقسم
│   ├── 2.7 validation.py            # تحقق بيانات محلي للقسم
│   ├── 2.8 api_interface.py         # واجهة برمجية للتكامل مع باقي النظام
│   ├── 2.9 event_bus.py             # نظام أحداث داخلي لتبادل الإشعارات والبيانات
│   ├── 2.10 integration_protocols/  # بروتوكولات التكامل مع الأنظمة الخارجية
│   │   ├── 2.10.1 rest_api_protocol.py
│   │   ├── 2.10.2 websocket_protocol.py
│   │   ├── 2.10.3 custom_integration_adapter.py
│   ├── 2.11 monitoring.py           # مراقبة أداء القسم وتسجيل الأحداث
│   ├── 2.12 extensibility.md        # توثيق نقاط التوسع والتكامل مع الأقسام الأخرى
│   # تم إضافة ملفات ونقاط تكامل عملية لرفع كفاءة القسم وزيادة قابلية التوسع والتكامل مع باقي النظام
│   └── 2.13 @integration.py
│
├── 3. Data Cleaning & Signal Processing/
│   ├── 3.1 Clean.py
│   ├── 3.2 MDPS.md
│   ├── 3.3 __init__.py
│   ├── 3.4 config.py
│   ├── 3.5 contextual_structural_annotation.py
│   ├── 3.6 data_quality_assurance.py
│   ├── 3.7 data_quality_monitoring.py
│   ├── 3.8 main.py
│   ├── 3.9 noise_signal_treatment.py
│   ├── 3.10 temporal_structural_alignment.py
│   ├── 3.11 data_manager.py         # إدارة بيانات محلية للقسم
│   ├── 3.12 validation.py           # تحقق بيانات محلي للقسم
│
├── 4. Preprocessing & Feature Engineering/
│   ├── 4.1 indicators/
│   │   ├── 4.1.1 technical_indicators.py
│   │   ├── 4.1.2 momentum_calculator.py
│   │   ├── 4.1.3 trend_strength_analyzer.py
│   │   ├── 4.1.4 volatility_band_mapper.py
│   │   ├── 4.1.5 ratio_spread_calculator.py
│   │   ├── 4.1.6 cycle_strength_analyzer.py
│   │   ├── 4.1.7 relative_position_encoder.py
│   │   ├── 4.1.8 price_action_density_mapper.py
│   │   ├── 4.1.9 microstructure_feature_extractor.py
│   │   ├── 4.1.10 market_depth_analyzer.py
│   ├── 4.2 encoders/
│   │   ├── 4.2.1 time_of_day_encoder.py
│   │   ├── 4.2.2 session_tracker.py
│   │   ├── 4.2.3 trend_context_tagger.py
│   │   ├── 4.2.4 volatility_spike_marker.py
│   │   ├── 4.2.5 cycle_phase_encoder.py
│   │   ├── 4.2.6 market_regime_classifier.py
│   │   ├── 4.2.7 volatility_regime_tagger.py
│   │   ├── 4.2.8 temporal_encoders.py
│   ├── 4.3 multi_scale/
│   │   ├── 4.3.1 multi_timeframe_feature_merger.py
│   │   ├── 4.3.2 lag_feature_engine.py
│   │   ├── 4.3.3 rolling_window_statistics.py
│   │   ├── 4.3.4 rolling_statistics_calculator.py
│   │   ├── 4.3.5 volume_tick_aggregator.py
│   │   ├── 4.3.6 pattern_window_slicer.py
│   │   ├── 4.3.7 feature_aggregator.py
│   │   ├── 4.3.8 candle_series_comparator.py
│   │   ├── 4.3.9 multi_scale_features.py
│   ├── 4.4 pattern_recognition/
│   │   ├── 4.4.1 candlestick_pattern_extractor.py
│   │   ├── 4.4.2 candlestick_shape_analyzer.py
│   │   ├── 4.4.3 pattern_encoder.py
│   │   ├── 4.4.4 price_cluster_mapper.py
│   │   ├── 4.4.5 pattern_sequence_embedder.py
│   │   ├── 4.4.6 pattern_recognition.py
│   ├── 4.5 feature_processing/
│   │   ├── 4.5.1 feature_generator.py
│   │   ├── 4.5.2 feature_aggregator.py
│   │   ├── 4.5.3 normalization_scaling_tools.py
│   │   ├── 4.5.4 correlation_filter.py
│   │   ├── 4.5.5 feature_selector.py
│   │   ├── 4.5.6 feature_processing.py
│   ├── 4.6 sequence_modeling/
│   │   ├── 4.6.1 sequence_constructor.py
│   │   ├── 4.6.2 temporal_encoder.py
│   │   ├── 4.6.3 sequence_modeling.py
│   ├── 4.7 versioning/
│   │   ├── 4.7.1 feature_version_control.py
│   │   ├── 4.7.2 feature_importance_tracker.py
│   │   ├── 4.7.3 auto_feature_selector.py
│   ├── 4.8 feature_monitoring.py
│   ├── 4.9 __init__.py
│   ├── 4.10 data_manager.py         # إدارة بيانات محلية للقسم
│   ├── 4.11 validation.py           # تحقق بيانات محلي للقسم
│
├── 5. Market_Context_Structural_Analysis/
│   ├── 5.1 key_zones_levels/
│   │   ├── 5.1.1 order_block_identifier.py
│   │   ├── 5.1.2 poi_tagger.py
│   │   ├── 5.1.3 supply_demand_zones.py
│   │   ├── 5.1.4 support_resistance_detector.py
│   ├── 5.2 liquidity_volume_structure/
│   │   ├── 5.2.1 fair_value_gap_detector.py
│   │   ├── 5.2.2 liquidity_gap_mapper.py
│   │   ├── 5.2.3 volume_profile_analyzer.py
│   │   ├── 5.2.4 vwap_band_generator.py
│   ├── 5.3 real_time_market_context/
│   │   ├── 5.3.1 liquidity_volatility_context_tags.py
│   │   ├── 5.3.2 market_state_generator.py
│   ├── 5.4 trend_structure_market_regime/
│   │   ├── 5.4.1 bos_detector.py
│   │   ├── 5.4.2 market_regime_classifier.py
│   │   ├── 5.4.3 mss_detector.py
│   │   ├── 5.4.4 peak_trough_detector.py
│   │   ├── 5.4.5 swing_high_low_labeler.py
│   │   ├── 5.4.6 trendline_channel_mapper.py
│   ├── 5.5 __init__.py
│   ├── 5.6 data_manager.py          # إدارة بيانات محلية للقسم
│   ├── 5.7 validation.py            # تحقق بيانات محلي للقسم
│
├── 6. Advanced Chart Analysis Tools/
│   ├── 6.1 advanced_indicators/
│   │   ├── 6.1.1 __init__.py
│   │   ├── 6.1.2 ichimoku_analyzer.py
│   │   ├── 6.1.3 supertrend_extractor.py
│   ├── 6.2 chart_pattern_detection/
│   │   ├── 6.2.1 __init__.py
│   │   ├── 6.2.2 chart_pattern_recognizer.py
│   │   ├── 6.2.3 fractal_pattern_detector.py
│   │   ├── 6.2.4 trend_channel_mapper.py
│   │   ├── 6.2.5 wolfe_wave_detector.py
│   ├── 6.3 elliott_wave_tools/
│   │   ├── 6.3.1 __init__.py
│   │   ├── 6.3.2 elliott_wave_analyzer.py
│   │   ├── 6.3.3 impulse_correction_classifier.py
│   ├── 6.4 fibonacci_geometric_tools/
│   │   ├── 6.4.1 __init__.py
│   │   ├── 6.4.2 fibonacci_toolkit.py
│   │   ├── 6.4.3 gann_fan_analyzer.py
│   ├── 6.5 harmonic_pattern_tools/
│   │   ├── 6.5.1 __init__.py
│   │   ├── 6.5.2 harmonic_pattern_identifier.py
│   │   ├── 6.5.3 harmonic_scanner.py
│   ├── 6.6 pattern_signal_fusion/
│   │   ├── 6.6.1 __init__.py
│   │   ├── 6.6.2 confidence_weighting_engine.py
│   │   ├── 6.6.3 pattern_signal_aggregator.py
│   ├── 6.7 price_action_annotators/
│   │   ├── 6.7.1 __init__.py
│   │   ├── 6.7.2 price_action_annotator.py
│   │   ├── 6.7.3 trend_context_tagger.py
│   ├── 6.8 support_resistance_tools/
│   │   ├── 6.8.1 __init__.py
│   │   ├── 6.8.2 pivot_point_tracker.py
│   │   ├── 6.8.3 supply_demand_identifier.py
│   │   ├── 6.8.4 support_resistance_finder.py
│   │   ├── 6.8.5 volume_profile_mapper.py
│   ├── 6.9 __init__.py
│   ├── 6.10 data_manager.py         # إدارة بيانات محلية للقسم
│   ├── 6.11 validation.py           # تحقق بيانات محلي للقسم
│
├── 7. Labeling & Target Engineering/
│   ├── 7.1 label_quality_assessment/
│   │   ├── 7.1.1 init.py
│   │   ├── 7.1.2 label_consistency_analyzer.py
│   │   ├── 7.1.3 label_noise_detector.py
│   ├── 7.2 label_transformers/
│   │   ├── 7.2.1 candle_direction_labeler.py
│   │   ├── 7.2.2 init.py
│   │   ├── 7.2.3 threshold_labeler.py
│   ├── 7.3 target_generators/
│   │   ├── 7.3.1 future_return_calculator.py
│   │   ├── 7.3.2 init.py
│   │   ├── 7.3.3 profit_zone_tagger.py
│   │   ├── 7.3.4 risk_reward_labeler.py
│   ├── 7.4 __init__.py
│   ├── 7.5 data_manager.py          # إدارة بيانات محلية للقسم
│   ├── 7.6 validation.py            # تحقق بيانات محلي للقسم
│
├── 8. Prediction Engine (MLDL Models)/
│   ├── 8.1 cnn_models/
│   │   ├── 8.1.1 autoencoder_feature_extractor.py
│   │   ├── 8.1.2 candle_image_encoder.py
│   │   ├── 8.1.3 cnn_signal_extractor.py
│   ├── 8.2 ensemble_models/
│   │   ├── 8.2.1 automation in trading systems.py
│   │   ├── 8.2.2 hybrid_ensemble.py
│   │   ├── 8.2.3 model_selector.py
│   │   ├── 8.2.4 signal_fusion.py
│   ├── 8.3 model_management/
│   │   ├── 8.3.1 drift_alerting.py
│   │   ├── 8.3.2 retraining_scheduler.py
│   │   ├── 8.3.3 version_control.py
│   ├── 8.4 reinforcement_learning/
│   │   ├── 8.4.1 environment_simulator.py
│   │   ├── 8.4.2 policy_evaluator.py
│   │   ├── 8.4.3 policy_gradient.py
│   │   ├── 8.4.4 strategy_optimizer.py
│   ├── 8.5 sequence_models/
│   │   ├── 8.5.1 attention_rnn.py
│   │   ├── 8.5.2 drift_detector.py
│   │   ├── 8.5.3 gru_sequence_model.py
│   │   ├── 8.5.4 informer_transformer.py
│   │   ├── 8.5.5 lstm_predictor.py
│   │   ├── 8.5.6 online_learning.py
│   ├── 8.6 traditional_ml/
│   │   ├── 8.6.1 __init__.py
│   │   ├── 8.6.2 cross_validation.py
│   │   ├── 8.6.3 random_forest_predictor.py
│   │   ├── 8.6.4 sklearn_pipeline.py
│   │   ├── 8.6.5 xgboost_classifier.py
│   ├── 8.7 training_utils/
│   │   ├── 8.7.1 hyperparameter_tuner.py
│   │   ├── 8.7.2 meta_learner_optimizer.py
│   │   ├── 8.7.3 model_explainer.py
│   │   ├── 8.7.4 performance_tracker.py
│   ├── 8.8 transformer_models/
│   │   ├── 8.8.1 meta_learner_optimizer.py
│   │   ├── 8.8.2 transformer_integrator.py
│   ├── 8.9 __init__.py
│   ├── 8.10 data_manager.py         # إدارة بيانات محلية للقسم
│   ├── 8.11 validation.py           # تحقق بيانات محلي للقسم
│
├── 9. Strategy & Decision Layer/
│   ├── 9.1 risk_management/
│   │   ├── 9.1.1 __init__.py
│   │   ├── 9.1.2 position_sizer.py
│   │   ├── 9.1.3 risk_manager.py
│   │   ├── 9.1.4 stop_target_generator.py
│   ├── 9.2 signal_validation/
│   │   ├── 9.2.1 __init__.py
│   │   ├── 9.2.2 confidence_scorer.py
│   │   ├── 9.2.3 direction_filter.py
│   │   ├── 9.2.4 signal_validator.py
│   ├── 9.3 simulation_analysis/
│   │   ├── 9.3.1 backtest_optimizer.py
│   │   ├── 9.3.2 execution_delay_emulator.py
│   │   ├── 9.3.3 feedback_loop.py
│   │   ├── 9.3.4 init.py
│   │   ├── 9.3.5 post_trade_analyzer.py
│   │   ├── 9.3.6 slippage_simulator.py
│   │   ├── 9.3.7 trade_simulator.py
│   │   ├── 9.3.8 transaction_cost_modeler.py
│   ├── 9.4 strategy_selection/
│   │   ├── 9.4.1 __init__.py
│   │   ├── 9.4.2 dynamic_selector.py
│   │   ├── 9.4.3 rule_based_system.py
│   │   ├── 9.4.4 strategy_selector.py
│   ├── 9.5 timing_execution/
│   │   ├── 9.5.1 init.py
│   │   ├── 9.5.2 timing_optimizer.py
│   ├── 9.6 main.py
│   ├── 9.7 __init__.py
│   ├── 9.8 data_manager.py          # إدارة بيانات محلية للقسم
│   ├── 9.9 validation.py            # تحقق بيانات محلي للقسم
│
├── 10. trading_ui/   ← System UI
│   ├── 10.1 config/                        # UI configuration files
│   ├── 10.2 core/                          # Core logic and controllers
│   │   ├── 10.2.1 data_manager.py            # Data management for UI
│   │   ├── 10.2.2 event_system.py            # Event handling and communication
│   │   ├── 10.2.3 market_data.py             # Market data processing for UI
│   │   ├── 10.2.4 mdps_controller.py         # Connects UI to MDPS system
│   │   └── 10.2.5 init.py
│   ├── 10.3 data/                          # Data storage and models
│   │   ├── 10.3.1 cache.py                   # Data caching
│   │   ├── 10.3.2 database.py                # UI database management
│   │   ├── 10.3.3 models/                    # Data models for UI
│   │   └── 10.3.4 init.py
│   ├── 10.4 services/                      # Service layer for UI
│   │   ├── 10.4.1 data_service.py            # Data fetching service
│   │   ├── 10.4.2 notification_service.py    # Notification and alerts
│   │   ├── 10.4.3 trading_service.py         # Trading operations service
│   │   └── 10.4.4 init.py
│   ├── 10.5 tests/                         # UI tests
│   ├── 10.6 ui/                            # UI components and views
│   │   ├── 10.6.1 main_window.py             # Main application window
│   │   ├── 10.6.2 resources/                 # UI resources (icons, images)
│   │   ├── 10.6.3 utils/                     # UI helper functions
│   │   ├── 10.6.4 views/                     # Different UI views/pages
│   │   ├── 10.6.5 widgets/                   # UI widgets (tables, buttons)
│   │   └── 10.6.6 init.py
│   ├── 10.7 utils/                         # General utilities for UI
│   │   ├── 10.7.1 constants.py               # UI constants
│   │   ├── 10.7.2 helpers.py                 # Helper functions
│   │   ├── 10.7.3 logger.py                  # Logging for UI
│   │   └── 10.7.4 init.py
│   ├── 10.8 init.py                        # UI initialization
│   ├── 10.9 main.py                        # UI entry point
│   └── 10.10 requirements.txt               # UI dependencies
│   # واجهة المستخدم تعتمد على البيانات من الأقسام الأخرى عبر واجهات التكامل فقط
│
├── 11. config.py      ← Central configuration
├── 12. main.py        ← Main entry point
├── 13. run_mdps.py    ← System runner script
├── 14. @orchestrator.py ← Tools orchestrator
-├── 15. @database.py    ← Central data management
-├── 16. @logging.py     ← Logging management
-├── 17. @error_handling.py ← Error management
│   # تم إلغاء وحدات الإدارة المركزية للبيانات والتحقق
├── 18. README.md      ← System documentation
├── 19. requirements.txt
├── 20. @tests/         ← System tests
└── 21. setup.py
```

ملاحظات:
- كل قسم يحتوي الآن على ملفات إدارة بيانات (data_manager.py) وملفات تحقق بيانات (validation.py) خاصة به.
- تم إلغاء وحدات الإدارة المركزية للبيانات والتحقق مثل @database.py و @logging.py و @error_handling.py.
- واجهة المستخدم تعتمد فقط على واجهات التكامل مع الأقسام الأخرى ولا تدير البيانات بشكل مركزي.
- يمكن إضافة بروتوكولات أو واجهات تكامل بين الأقسام عند الحاجة لتبادل البيانات.
