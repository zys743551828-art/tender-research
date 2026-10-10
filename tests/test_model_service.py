"""使用夹具验证扩展和路由边界；不调用任何外部厂商。"""
import unittest
from pathlib import Path
from unittest.mock import Mock

from tender_research.model_service import ProviderError, ProviderRegistry, available_adapters, create_provider


class ProviderRoutingTests(unittest.TestCase):
    def test_current_adapter_only_deepseek(self):
        self.assertEqual(available_adapters(), ('deepseek',))

    def test_unknown_provider_cannot_fall_back_to_deepseek(self):
        with self.assertRaises(ProviderError) as caught:
            create_provider(Path('/synthetic/no-credential'), 'unconfigured-fixture')
        self.assertEqual(caught.exception.category, 'unsupported_provider')

    def test_registration_is_lazy(self):
        factory = Mock()
        registry = ProviderRegistry()
        registry.register('fixture', factory)
        factory.assert_not_called()

    def test_adapter_addition_preserves_existing_route(self):
        registry = ProviderRegistry()
        first = Mock(provider_id='first', complete=Mock(return_value={'provider': 'first', 'output': {'answer': 1}}))
        second = Mock(provider_id='second', complete=Mock(return_value={'provider': 'second', 'output': {'answer': 2}}))
        registry.register('first', lambda root: first)
        registry.register('second', lambda root: second)
        result = registry.create('first', Path('/fixture')).complete([], 'fixture-model')
        self.assertEqual(result['provider'], 'first')
        second.complete.assert_not_called()

    def test_duplicate_adapter_cannot_replace_existing_route(self):
        registry = ProviderRegistry(); registry.register('fixture', lambda root: Mock(provider_id='fixture'))
        with self.assertRaises(ValueError):
            registry.register('fixture', lambda root: Mock(provider_id='replacement'))

    def test_mismatched_adapter_identity_rejected(self):
        registry = ProviderRegistry(); registry.register('first', lambda root: Mock(provider_id='second'))
        with self.assertRaises(ProviderError) as caught:
            registry.create('first', Path('/fixture'))
        self.assertEqual(caught.exception.category, 'provider_identity_mismatch')


if __name__ == '__main__':
    unittest.main()
