import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Button } from '../../src/components/common/Button';

describe('Button Component', () => {
  it('renders correctly with title', () => {
    const { getByText } = render(<Button title="Test Button" onPress={() => {}} />);
    expect(getByText('Test Button')).toBeTruthy();
  });

  it('calls onPress when pressed', () => {
    const onPressMock = jest.fn();
    const { getByText } = render(<Button title="Test Button" onPress={onPressMock} />);
    
    fireEvent.press(getByText('Test Button'));
    
    expect(onPressMock).toHaveBeenCalledTimes(1);
  });

  it('is disabled when loading', () => {
    const onPressMock = jest.fn();
    const { getByText } = render(
      <Button title="Test Button" onPress={onPressMock} loading={true} />
    );
    
    const button = getByText('Test Button').parent;
    expect(button?.props.accessibilityState?.disabled).toBe(true);
  });
});

